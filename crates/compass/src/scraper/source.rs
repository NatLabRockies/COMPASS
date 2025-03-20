//! Scrapped document

use sha2::Digest;
use tokio::io::AsyncReadExt;
use tracing::{error, trace, warn};

use crate::error::Result;

#[derive(Debug)]
/// Source document scrapped
pub(super) struct Source {
    name: String,
    hash: String,
    #[allow(dead_code)]
    origin: Option<String>,
    #[allow(dead_code)]
    access_time: Option<String>,
}

impl Source {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        trace!("Initializing database for Source");

        trace!("Creating table archive");
        // Store all individual documents scrapped
        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS archive_sequence START 1;
            CREATE TABLE IF NOT EXISTS archive (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('archive_sequence'),
              name TEXT NOT NULL,
              hash TEXT NOT NULL,
              origin TEXT,
              access_time TIMESTAMP,
              created_at TIMESTAMP NOT NULL DEFAULT NOW(),
              );",
        )?;

        trace!("Creating table source");
        // Register the target documents of each scrapping run
        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS source_sequence START 1;
            CREATE TABLE IF NOT EXISTS source (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('source_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              archive_lnk INTEGER REFERENCES archive(id) NOT NULL,
              );",
        )?;

        Ok(())
    }

    /// Open the source documents that were scrapped
    ///
    /// # Returns
    ///
    /// * A vector of source documents
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Vec<Source>> {
        trace!("Opening source documents");

        let path = root.as_ref().join("ordinance_files");
        if !path.exists() {
            error!("Missing source directory: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Source directory does not exist".to_string(),
            ));
        }

        trace!("Scanning source directory: {:?}", path);

        let mut sources = vec![];
        let mut inventory = tokio::fs::read_dir(path).await?;

        // Should we filter which files to process, such as only PDFs?
        // We probably will work with more types.
        while let Some(entry) = inventory.next_entry().await? {
            let path = entry.path();
            let metadata = entry.metadata().await?;
            let file_type = metadata.file_type();

            if file_type.is_file() {
                trace!("Processing ordinance file: {:?}", path);

                let checksum = checksum_file(&path).await?;
                let s = Source {
                    name: path.file_name().unwrap().to_string_lossy().to_string(),
                    hash: checksum,
                    origin: None,
                    access_time: None,
                };
                trace!("Identified a new source: {:?}", s);
                sources.push(s);
            } else if file_type.is_dir() {
                warn!(
                    "Ignoring unexpected directory in ordinance files: {:?}",
                    path
                );
            }
        }

        trace!("Found a total of {} source documents", sources.len());

        Ok(sources)
    }

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
