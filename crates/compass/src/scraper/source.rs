//! Scrapped document

use serde::Deserialize;
use sha2::Digest;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, trace, warn};

use crate::error::Result;

// An arbitrary limit (5MB) to protect against maliciously large JSON files
const MAX_JSON_FILE_SIZE: u64 = 5 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub(super) struct Source {
    pub(super) jurisdictions: Vec<Jurisdiction>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Jurisdiction {
    full_name: String,
    county: String,
    state: String,
    subdivision: Option<String>,
    jurisdiction_type: Option<String>,
    #[serde(alias = "FIPS")]
    /// FIPS
    fips: u32,
    found: bool,
    total_time: f64,
    total_time_string: String,
    documents: Option<Vec<Document>>,
}

#[derive(Deserialize, Debug)]
pub(super) struct Document {
    source: String,
    // Maybe use effective instead?
    ord_year: u16,
    ord_filename: String,
    num_pages: u16,
    checksum: String,
    #[allow(dead_code)]
    access_time: Option<String>,
}

impl Source {
    /// Initialize database for the Source context
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
            ord_year INTEGER,
            filename TEXT,
            num_pages INTEGER,
            checksum TEXT,
            access_time TIMESTAMP,
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
            fips INTEGER,
            found BOOLEAN,
            total_time REAL,
            total_time_string TEXT,
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

    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        debug!("Opening source documents");

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

        let mut inventory = tokio::fs::read_dir(path).await?;

        // Should we filter which files to process, such as only PDFs?
        // We probably will work with more types.
        while let Some(entry) = inventory.next_entry().await? {
            trace!("Processing entry: {:?}", entry);
            let path = entry.path();
            let ftype = entry.metadata().await?.file_type();
            if ftype.is_file() {
                trace!("Processing ordinance file: {:?}", path);

                let checksum = checksum_file(&path).await?;
                trace!("Checksum for file {:?}: {:?}", path, checksum);

                let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
                if known_sources.contains(&(file_name, checksum)) {
                    trace!("File {:?} matches known jurisdiction source", path);
                } else {
                    warn!("File {:?} doesn't match known sources", path);
                }
            } else if ftype.is_dir() {
                trace!(
                    "Ignoring unexpected directory in ordinance files: {:?}",
                    path
                );
            }
        }

        // trace!("Found a total of {} source documents", sources.len());

        Ok(jurisdictions)
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
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
                    (source, ord_year, filename, num_pages,
                      checksum)
                    VALUES (?, ?, ?, ?, ?)
                    RETURNING id",
                )?;

                for document in documents {
                    trace!("Inserting document: {:?}", document);
                    let did = stmt_archive
                        .query(duckdb::params![
                            document.source,
                            document.ord_year,
                            document.ord_filename,
                            document.num_pages,
                            document.checksum,
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
                  documents)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                dids.iter()
                    .map(|did| did.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            ])?;
        }
        Ok(())
    }

    /*
    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        trace!("Recording source documents on database");

        // What about return the number of rows inserted?

        /*
        let origin = match &self.origin {
            Some(origin) => origin,
            None => {
                trace!("Missing origin for document {}", &self.name);
                "NULL"
            }
        };
        let access_time = match &self.access_time {
            Some(time) => time,
            None => {
                trace!("Missing access time for document {}", &self.name);
                "NULL"
            }
        };
        */

        // Insert the source document into the database
        let source_id: u32 = conn.query_row(
            "INSERT INTO archive (name, hash) VALUES (?, ?) RETURNING id",
            [&self.name, &self.hash],
            |row| row.get(0),
        )?;
        trace!(
            "Inserted source document with id: {} -> {}",
            source_id, &self.name
        );
        conn.execute(
            "INSERT INTO source (bookkeeper_lnk, archive_lnk) VALUES (?, ?)",
            [commit_id.to_string(), source_id.to_string()],
        )?;
        trace!(
            "Linked source: commit ({}) -> archive ({})",
            commit_id, source_id
        );

        Ok(())
    }
    */
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
    let checksum = format!("sha256:{:x}", result);

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
                    "documents": [
                        {
                            "source": "https://example.com/sample_ordinance.pdf",
                            "ord_year": 2023,
                            "ord_filename": "sample_ordinance.pdf",
                            "num_pages": 10,
                            "checksum": "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
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
