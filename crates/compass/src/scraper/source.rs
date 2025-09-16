//! Scrapped documents
//!
//! A scrapping job saves the content source documents and some metadata
//! associated to those. This module provides the resources to parse that
//! information and store it in the database.
//!
//! It is expected that the outputs of the scrapping are stored in a
//! directory with:
//! - `jurisdictions.json`: A JSON file with information on the target
//!   jurisdictions, including the documents scrapped.
//! - `ordinance_files/` - A directory with the scrapped documents.

use serde::Deserialize;
use sha2::Digest;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, trace, warn};

use super::MAX_JSON_FILE_SIZE;
use crate::error::Result;

#[derive(Debug, Deserialize)]
/// A collection of target jurisdictions and related information
pub(super) struct Source {
    pub(super) jurisdictions: Vec<Jurisdiction>,
}

#[derive(Debug, Deserialize)]
/// A jurisdiction and its metadata
pub(super) struct Jurisdiction {
    /// Full name of the jurisdiction, such as "Golden City, Colorado"
    full_name: String,
    /// County where the jurisdiction is located, such as "Jefferson County"
    county: String,
    /// State where the jurisdiction is located, such as "Colorado"
    state: String,
    /// Subdivision of the jurisdiction, if any, such as "Golden"
    subdivision: Option<String>,
    /// Type of jurisdiction, such as "city", "county", etc.
    jurisdiction_type: Option<String>,
    #[serde(alias = "FIPS")]
    /// Federal Information Processing Standards code for the jurisdiction
    fips: u64,
    /// Whether the jurisdiction was found during the scraping
    found: bool,
    /// Total time spent scraping the jurisdiction, in seconds
    total_time: f64,
    /// Total time spent scraping the jurisdiction, as a string
    total_time_string: String,
    /// Main jurisdiction website used for web crawling, if any, as a string
    jurisdiction_website: Option<String>,
    /// Whether the jurisdiction document was found using the custom compass website crawl
    compass_crawl: bool,
    /// Total cost to run the scraper, in $
    cost: Option<f64>,
    /// List of documents associated with the jurisdiction
    documents: Option<Vec<Document>>,
}

#[derive(Deserialize, Debug)]
/// Processed document
///
/// Represents a document target of the scraper and its metadata.
/// Although it is typically a PDF, it could be any sort of document,
/// such as plain text from a website.
pub(super) struct Document {
    /// Source of the document, such as a URL
    source: String,
    /// Day that the ordinance went into effect, such as 4
    effective_day: Option<u16>,
    /// Month that the ordinance went into effect, such as 27
    effective_month: Option<u16>,
    /// Year that the ordinance went into effect, such as 2023
    effective_year: Option<u16>,
    /// Filename of the ordinance document
    ord_filename: String,
    /// Number of pages in the ordinance document
    num_pages: u16,
    /// Checksum of the original raw document
    checksum: String,
    /// Whether the document is a PDF file
    is_pdf: bool,
    /// Whether the document text was parsed using OCR
    from_ocr: bool,
    #[allow(dead_code)]
    /// When the document was obtained, i.e. downloaded.
    access_time: Option<String>,
    /// N-gram score for extracting ordinance text
    ordinance_text_ngram_score: Option<f64>,
    /// N-gram score for extracting permitted uses text
    permitted_use_text_ngram_score: Option<f64>,
}

impl Source {
    /// Initialize database for the Source context
    ///
    /// # Arguments
    ///
    /// * `conn` - A reference to the DuckDB transaction to execute the SQL commands.
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        debug!("Initializing database for Source");

        trace!("Creating table archive");
        // Store all individual documents scrapped
        conn.execute_batch(
            r"
          CREATE SEQUENCE IF NOT EXISTS archive_sequence START 1;
          CREATE TABLE IF NOT EXISTS archive (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('archive_sequence'),
            source TEXT,
            effective_day INTEGER,
            effective_month INTEGER,
            effective_year INTEGER,
            filename TEXT,
            num_pages INTEGER,
            checksum TEXT,
            is_pdf BOOLEAN,
            from_ocr BOOLEAN,
            access_time TIMESTAMP,
            ordinance_text_ngram_score REAL,
            permitted_use_text_ngram_score REAL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            );",
        )?;

        trace!("Creating table source");
        conn.execute_batch(
            r"
          CREATE SEQUENCE IF NOT EXISTS source_sequence START 1;
          CREATE TABLE IF NOT EXISTS source (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('source_sequence'),
            bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
            full_name TEXT,
            county TEXT,
            state TEXT,
            subdivision TEXT,
            jurisdiction_type TEXT,
            fips UBIGINT,
            found BOOLEAN,
            total_time REAL,
            total_time_string TEXT,
            jurisdiction_website TEXT,
            compass_crawl BOOLEAN,
            cost REAL,
            documents TEXT,
            archive_lnk INTEGER REFERENCES archive(id),
            );",
        )?;

        trace!("Database ready for Source");
        Ok(())
    }

    fn from_json(content: &str) -> Result<Self> {
        trace!("Parsing sources from JSON: {:?}", content);

        let source = serde_json::from_str(content).unwrap();
        Ok(source)
    }

    /// Open a Source collection from a scrapped output directory
    ///
    /// The Source collects all the documents scrapped and related metadata.
    /// This method verifies the expected contents and parses the relevant
    /// information.
    ///
    /// Currently, it expects:
    /// * `jurisdictions.json` - A JSON file containing jurisdiction data.
    /// * `ordinance_files` - A directory containing the files scrapped.
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory where the scrapped output is located.
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        debug!("Opening source documents from {:?}", root.as_ref());

        trace!("Opening jurisdictions collection");

        let path = root.as_ref().join("jurisdictions.json");
        if !path.exists() {
            error!("Missing jurisdictions.json file");
            return Err(crate::error::Error::Undefined(
                "Missing jurisdictions.json file".to_string(),
            ));
        }

        trace!("Identified jurisdictions.json file");

        let file_size = tokio::fs::metadata(&path).await?.len();
        if file_size > MAX_JSON_FILE_SIZE {
            error!("Jurisdictions file too large: {:?}", file_size);
            return Err(crate::error::Error::Undefined(
                "jurisdictions.json file is too large".to_string(),
            ));
        }

        let content = tokio::fs::read_to_string(path).await?;
        let jurisdictions = Self::from_json(&content)?;
        trace!("Jurisdictions loaded: {:?}", jurisdictions);

        // ========================

        let known_sources = jurisdictions
            .jurisdictions
            .iter()
            .filter_map(|j| j.documents.as_ref())
            .flatten()
            .map(|d| (d.ord_filename.clone(), d.checksum.clone()))
            .collect::<Vec<_>>();
        trace!("Known sources: {:?}", known_sources);

        let path = root.as_ref().join("ordinance_files");
        if !path.exists() {
            error!("Missing source directory: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Source directory does not exist".to_string(),
            ));
        }

        trace!("Scanning source directory: {:?}", path);

        let mut walker = tokio::fs::read_dir(&path).await?;
        let mut jobs = tokio::task::JoinSet::new();
        while let Some(entry) = walker.next_entry().await? {
            trace!("Spawning job for entry: {:?}", entry.path());
            jobs.spawn(async move { File::new(entry.path()).await });
        }
        trace!("Waiting for all jobs to complete");
        let inventory = jobs.join_all().await;
        trace!("Inventory of files: {:?}", inventory);
        debug!("Finished reading {} source documents", inventory.len());

        for file in inventory {
            match file {
                Ok(file) => {
                    let (file_name, checksum) = (file.filename, file.checksum);
                    if known_sources.contains(&(file_name, checksum)) {
                        trace!("File {:?} matches known jurisdiction source", file.path);
                    } else {
                        warn!("File {:?} doesn't match known sources", file.path);
                    }
                }
                Err(e) => {
                    error!("Error processing file: {:?}", e);
                }
            }
        }

        Ok(jurisdictions)
    }

    /// Record the Source collection in the database
    ///
    /// While the information (metadata) of the source documents are
    /// recorded in the database, the actual documents are not stored.
    ///
    /// # Arguments
    ///
    /// * `conn` - A reference to the DuckDB transaction to execute the SQL commands.
    pub(super) fn record(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        debug!("Recording jurisdictions on database");

        for jurisdiction in &self.jurisdictions {
            trace!("Inserting jurisdiction: {:?}", jurisdiction);

            let mut dids = Vec::new();
            if let Some(documents) = &jurisdiction.documents {
                // Replace this by a query, if not found already in the database, insert and return
                // the id.
                let mut stmt_archive = conn.prepare(
                    r"
                    INSERT INTO archive
                    (source, effective_day, effective_month, effective_year, filename, num_pages,
                      checksum, is_pdf, from_ocr, ordinance_text_ngram_score, permitted_use_text_ngram_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id",
                )?;

                for document in documents {
                    trace!("Inserting document: {:?}", document);
                    let did = stmt_archive
                        .query(duckdb::params![
                            document.source,
                            document.effective_day,
                            document.effective_month,
                            document.effective_year,
                            document.ord_filename,
                            document.num_pages,
                            document.checksum,
                            document.is_pdf,
                            document.from_ocr,
                            document.ordinance_text_ngram_score,
                            document.permitted_use_text_ngram_score,
                        ])?
                        .next()
                        .unwrap()
                        .unwrap()
                        .get::<_, i64>(0)
                        .unwrap();
                    dids.push(did);
                }
                trace!("Inserted documents' ids: {:?}", dids);
            } else {
                trace!("No documents found for jurisdiction: {:?}", jurisdiction);
            }

            let mut stmt_source = conn.prepare(
                r"
                INSERT INTO source
                (bookkeeper_lnk, full_name, county, state,
                  subdivision, jurisdiction_type, fips,
                  found, total_time, total_time_string,
                  jurisdiction_website, compass_crawl, cost, documents)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )?;
            stmt_source.execute(duckdb::params![
                commit_id,
                jurisdiction.full_name,
                jurisdiction.county,
                jurisdiction.state,
                jurisdiction.subdivision,
                jurisdiction.jurisdiction_type,
                jurisdiction.fips,
                jurisdiction.found,
                jurisdiction.total_time,
                jurisdiction.total_time_string,
                jurisdiction.jurisdiction_website,
                jurisdiction.compass_crawl,
                jurisdiction.cost,
                dids.iter()
                    .map(|did| did.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            ])?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct File {
    path: std::path::PathBuf,
    filename: String,
    checksum: String,
}

impl File {
    async fn new<P: AsRef<std::path::Path> + std::fmt::Debug>(path: P) -> Result<Self> {
        debug!("Processing ordinance file: {:?}", path.as_ref());

        let metadata = tokio::fs::metadata(&path).await?;
        if !metadata.is_file() {
            warn!("Expected to be a file. Ignoring: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Expected a file, but found a directory or other type".to_string(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        let checksum = checksum_file(&path).await?;
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        Ok(Self {
            path,
            filename,
            checksum,
        })
    }
}
/// Calculate the checksum of a local file
///
/// # Returns
///
/// * The checksum of the file with a tag indicating the algorithm used
///   (e.g. `sha256:...`)
async fn checksum_file<P: AsRef<std::path::Path>>(path: P) -> Result<String> {
    trace!("Calculating checksum for {:?}", path.as_ref());
    let mut hasher = sha2::Sha256::new();

    let f = tokio::fs::File::open(&path).await?;
    let mut reader = tokio::io::BufReader::new(f);
    let mut buffer: [u8; 1024] = [0; 1024];
    while let Ok(n) = reader.read(&mut buffer).await {
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let result = hasher.finalize();
    let checksum = format!("sha256:{result:x}");

    trace!("Checksum for {:?}: {}", path.as_ref(), checksum);
    Ok(checksum)
}

#[cfg(test)]
/// Samples of source documents
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    /// Create a sample source document
    ///
    /// For now, limited to jurisdictions.json and missing sample
    /// documents.
    pub(crate) fn as_text() -> String {
        r#"
        {
            "jurisdictions": [
                {
                    "full_name": "Sample Jurisdiction",
                    "county": "Sample County",
                    "state": "Sample State",
                    "subdivision": null,
                    "jurisdiction_type": null,
                    "FIPS": 12345,
                    "found": true,
                    "total_time": 3.14,
                    "total_time_string": "0::0::03.14",
                    "jurisdiction_website": null,
                    "compass_crawl": false,
                    "documents": [
                        {
                            "source": "https://example.com/sample_ordinance.pdf",
                            "effective_month": 4,
                            "effective_year": 2023,
                            "effective_day": null,
                            "ord_filename": "sample_ordinance.pdf",
                            "num_pages": 10,
                            "checksum": "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                            "is_pdf": true,
                            "from_ocr": false,
                            "ordinance_text_ngram_score": 0.95,
                            "permitted_use_text_ngram_score": null
                        }
                    ]
                }
            ]
        }
        "#.to_string()
    }

    pub(crate) fn as_file<P: AsRef<std::path::Path>>(path: P) -> Result<std::fs::File> {
        let mut file = std::fs::File::create(path)?;
        write!(file, "{}", as_text())?;
        Ok(file)
    }
}
