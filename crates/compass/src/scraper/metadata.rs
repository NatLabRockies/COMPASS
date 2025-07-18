//! Parse and handle the Scrapper configuration information
//!
//! The setup used to run the scraper is saved together with the output.
//! This module provides the support to work with that information, from
//! validating and parsing to loading it in the database.

use std::collections::HashMap;

use tracing::debug;

use crate::error::Result;

// An arbitrary limit to protect against maliciously large JSON files
const MAX_JSON_FILE_SIZE: u64 = 5_000_000;

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Configuration used to run the scraper
pub(super) struct Metadata {
    username: String,
    versions: HashMap<String, String>,
    technology: String,
    models: Vec<LLMMetadata>,
    time_start_utc: String,
    time_end_utc: String,
    total_time: f64,
    total_time_string: String,
    num_jurisdictions_searched: u16,
    num_jurisdictions_found: u16,
    cost: Option<f64>,
    manifest: HashMap<String, String>,

    #[serde(flatten)]
    pub(crate) extra: HashMap<String, serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Configuration used to run the scraper
pub(super) struct LLMMetadata {
    name: String,
    llm_call_kwargs: Option<HashMap<String, serde_json::Value>>,
    llm_service_rate_limit: u64,
    text_splitter_chunk_size: u32,
    text_splitter_chunk_overlap: u32,
    client_type: String,
    tasks: Vec<String>,

    #[serde(flatten)]
    pub(crate) extra: HashMap<String, serde_json::Value>,
}

impl Metadata {
    /// Initialize the database to support Metadata
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        tracing::trace!("Initializing database for Metadata");
        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS scraper_metadata_sequence START 1;
            CREATE TABLE IF NOT EXISTS scraper_metadata (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('scraper_metadata_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              username TEXT,
              versions TEXT,
              technology TEXT,
              time_start_utc TEXT,
              time_end_utc TEXT,
              total_time REAL,
              num_jurisdictions_searched INTEGER,
              num_jurisdictions_found INTEGER,
              cost REAL,
              manifest TEXT,
              extra TEXT,
            );

            CREATE SEQUENCE IF NOT EXISTS llm_config_sequence START 1;
            CREATE TABLE IF NOT EXISTS llm_config (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('llm_config_sequence'),
              metadata_lnk INTEGER REFERENCES scraper_metadata(id) NOT NULL,
              name TEXT,
              llm_call_kwargs TEXT,
              llm_service_rate_limit INTEGER,
              text_splitter_chunk_size INTEGER,
              text_splitter_chunk_overlap INTEGER,
              client_type TEXT,
              tasks TEXT,
              extra TEXT,
            );",
        )?;

        tracing::trace!("Database ready for Metadata");
        Ok(())
    }

    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        debug!("Opening Metadata from {:?}", root.as_ref());

        let path = root.as_ref().join("meta.json");
        if !path.exists() {
            tracing::error!("Missing metadata file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing metadata file".to_string(),
            ));
        }

        tracing::trace!("Identified Metadata at {:?}", path);

        // These JSON files are expected to be tiny, so protect against
        // huge files that probably means some mistake.
        let filesize = tokio::fs::metadata(&path).await?.len();
        if filesize > MAX_JSON_FILE_SIZE {
            tracing::error!("Metadata file too large: {:?}", filesize);
            return Err(crate::error::Error::Undefined(
                "Metadata file too large".to_string(),
            ));
        }

        let content = tokio::fs::read_to_string(path).await?;
        let metadata = Self::from_json(&content)?;
        tracing::trace!("Metadata loaded: {:?}", metadata);

        Ok(metadata)
    }

    #[allow(dead_code)]
    /// Extract the configuration from a JSON string
    pub(super) fn from_json(json: &str) -> Result<Self> {
        tracing::trace!("Parsing Metadata from JSON: {:?}", json);
        let metadata: Metadata = serde_json::from_str(json).unwrap();
        Ok(metadata)
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        tracing::trace!("Writing Metadata to the database {:?}", self);
        let metadata_id: u32 = conn
            .query_row(
                r"INSERT INTO scraper_metadata
                     (bookkeeper_lnk, username, versions, technology,
                       time_start_utc, time_end_utc,
                       total_time, num_jurisdictions_searched,
                       num_jurisdictions_found, cost, manifest)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                duckdb::params![
                    commit_id,
                    self.username,
                    serde_json::to_string(&self.versions).unwrap(),
                    self.technology,
                    self.time_start_utc,
                    self.time_end_utc,
                    self.total_time,
                    self.num_jurisdictions_searched,
                    self.num_jurisdictions_found,
                    self.cost,
                    serde_json::to_string(&self.manifest).unwrap(),
                ],
                |row| row.get(0),
            )
            .expect("Failed to insert metadata");

        for llm_metadata in &self.models {
            tracing::trace!(
                "Writing metadata for {:?} to the database",
                llm_metadata.name
            );
            conn.execute(
                "INSERT INTO llm_config
                    (metadata_lnk, name, llm_call_kwargs, llm_service_rate_limit,
                     text_splitter_chunk_size, text_splitter_chunk_overlap,
                     client_type, tasks, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                duckdb::params![
                    metadata_id,
                    llm_metadata.name,
                    serde_json::to_string(&llm_metadata.llm_call_kwargs).unwrap(),
                    llm_metadata.llm_service_rate_limit,
                    llm_metadata.text_splitter_chunk_size,
                    llm_metadata.text_splitter_chunk_overlap,
                    llm_metadata.client_type,
                    serde_json::to_string(&llm_metadata.tasks).unwrap(),
                    serde_json::to_string(&llm_metadata.extra).unwrap(),
                ],
            )
            .expect("Failed to insert usage_event");
        }

        Ok(())
    }
}

#[cfg(test)]
/// Samples of scraper configuration to support tests
///
/// These samples should cover multiple versions of data models as this library evolves and it
/// should be accessible from other parts of the crate.
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    pub(crate) fn as_text_v1() -> String {
        r#"
    {
    "username": "ppinchuk",
    "versions": {
        "elm": "0.0.11",
        "compass": "0.1.1.dev17+gb569353.d20250304"
        },
    "technology": "wind",
    "models": [
        {
            "name": "gpt-4.1-mini",
            "llm_call_kwargs": {
                "temperature": 0,
                "seed": 42,
                "timeout": 300
            },
            "llm_service_rate_limit": 500000,
            "text_splitter_chunk_size": 10000,
            "text_splitter_chunk_overlap": 500,
            "client_type": "azure",
            "tasks": ["default"]
        }
    ],
    "time_start_utc": "2025-03-04T05:10:52.266550+00:00",
    "time_end_utc": "2025-03-04T05:19:49.767500+00:00",
    "total_time": 537.5009291959941,
    "total_time_string": "0:08:57.500929",
    "num_jurisdictions_searched": 10,
    "num_jurisdictions_found": 7,
    "cost": 0.17406639999999998,
    "manifest": {
        "LOG_DIR": "logs",
        "META_FILE": "meta.json"
        }
    }"#
        .to_string()
    }

    pub(crate) fn as_file<P: AsRef<std::path::Path>>(path: P) -> Result<std::fs::File> {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "{}", as_text_v1()).unwrap();
        Ok(f)
    }
}

#[cfg(test)]
mod test_scraper_metadata {
    use super::Metadata;
    use super::sample::as_text_v1;

    #[test]
    /// Load a Metadata from a JSON string
    fn parse_json() {
        let metadata = Metadata::from_json(&as_text_v1()).unwrap();

        assert_eq!(metadata.username, "ppinchuk");
        assert_eq!(metadata.num_jurisdictions_searched, 10);
    }
}
