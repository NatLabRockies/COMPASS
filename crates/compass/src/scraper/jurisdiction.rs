//! Parse and handle the scrapped jurisdiction information


use std::collections::HashMap;

use serde::Deserialize;
use tracing::{trace, warn, error};

use crate::Result;

// An arbitrary limit to protect against maliciously large JSON files
const MAX_JSON_FILE_SIZE: u64 = 5 * 1024 * 1024; // 10 MB

#[derive(Deserialize, Debug)]
pub(super) struct Collection {
    pub(super) jurisdictions: Vec<Jurisdiction>,
}

#[derive(Deserialize, Debug)]
pub(super) struct Jurisdiction {
    full_name: String,
    county: String,
    state: String,
    subdivision: Option<String>,
    jurisdiction_type: Option<String>,
    FIPS: u32,
    found: bool,
    total_time: f64,
    total_time_string: String,
    documents: Option<Vec<Document>>,
}

#[derive(Deserialize, Debug)]
pub(super) struct Document {
    source: String,
    ord_year: u16,
    ord_filename: String,
    num_pages: u16,
    checksum: String,
}

impl Collection {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        warn!("Initializing database for jurisdictions");

        // archive
        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS archive_sequence START 1;
            CREATE TABLE IF NOT EXISTS archive (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('archive_sequence'),
              source TEXT,
              origin TEXT,
              ord_year INTEGER,
              ord_filename TEXT,
              name TEXT,
              num_pages INTEGER,
              checksum TEXT,
              hash TEXT,
              access_time TIMESTAMP,
              created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            );",
        )?;

        warn!("Database ready for jurisdictions");
        Ok(())
    }

    fn from_json(content: &str) -> Result<Self> {
        warn!("Parsing jurisdictions from JSON: {:?}", content);
        let collection: Collection = serde_json::from_str(content).unwrap();
        Ok(collection)
    }

    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        warn!("Opening jurisdictions collection");

        let path = root.as_ref().join("jurisdictions.json");
        if !path.exists() {
            error!("Missing jurisdictions.json file");
            return Err(crate::error::Error::Undefined(
                "Missing jurisdictions.json file".to_string(),
            ));
        }

        warn!("Identified jurisdictions.json file");

        let file_size = tokio::fs::metadata(&path).await?.len();
        if file_size > MAX_JSON_FILE_SIZE {
            error!("Jurisdictions file too large: {:?}", file_size);
            return Err(crate::error::Error::Undefined(
                "jurisdictions.json file is too large".to_string(),
            ));
        }

        let content = tokio::fs::read_to_string(path).await?;
        let jurisdictions = Self::from_json(&content)?;
        warn!("Jurisdictions loaded: {:?}", jurisdictions);

        Ok(jurisdictions)
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        warn!("Recording jurisdictions on database");

        for jurisdiction in &self.jurisdictions {
            warn!("Inserting jurisdiction: {:?}", jurisdiction);
            if let Some(documents) = &jurisdiction.documents {
                // Replace this by a query, if not found already in the database, insert and return
                // the id.
                let mut stmt_archive = conn.prepare(
                    r"
                    INSERT INTO archive
                    (source, ord_year, ord_filename, num_pages,
                      checksum)
                    VALUES (?, ?, ?, ?, ?)
                    RETURNING id",
                )?;

                let mut dids = Vec::new();
                for document in documents {
                    warn!("Inserting document: {:?}", document);
                    let did = stmt_archive.query(duckdb::params! [
                        document.source,
                        document.ord_year,
                        document.ord_filename,
                        document.num_pages,
                        document.checksum,
                    ])?.next().unwrap().unwrap().get::<_, i64>(0).unwrap();
                    dids.push(did);
                }
                warn!("Inserted documents' ids: {:?}", dids);

            } else {
                warn!("No documents found for jurisdiction: {:?}", jurisdiction);
            }
        }
        Ok(())

    }
}
