//! Parse and handle the Scrapper configuration information
//!
//! The setup used to run the scrapper is saved together with the output.
//! This module provides the support to work with that information, from
//! validating and parsing to loading it in the database.

use std::io::Read;

use crate::error::Result;

#[allow(dead_code)]
#[derive(Debug)]
/// Configuration used to run the scrapper
pub(crate) struct ScrapperConfig {
    pub(crate) model: String,
    pub(crate) llm_service_rate_limit: u64,
    pub(crate) extra: String,
}

impl ScrapperConfig {
    /// Initialize the database to support ScrapperConfig
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        tracing::trace!("Initializing database for ScrapperConfig");
        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS scrapper_config_sequence START 1;
            CREATE TABLE IF NOT EXISTS scrapper_config (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('scrapper_config_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              model TEXT NOT NULL,
              llm_service_rate_limit INTEGER,
              extra TEXT,
            );",
        )?;

        tracing::trace!("Database ready for ScrapperConfig");
        Ok(())
    }

    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        tracing::trace!("Opening ScrapperConfig from {:?}", root.as_ref());

        let path = root.as_ref().join("config.json");
        if !path.exists() {
            tracing::error!("Missing configuration file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing configuration file".to_string(),
            ));
        }

        tracing::trace!("Identified ScrapperConfig at {:?}", path);

        let file = std::fs::File::open(path);
        let mut reader = std::io::BufReader::new(file.unwrap());
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer);

        let config = Self::from_json(&buffer)?;
        tracing::trace!("ScrapperConfig loaded: {:?}", config);

        Ok(config)
    }

    #[allow(dead_code)]
    /// Extract the configuration from a JSON string
    pub(super) fn from_json(json: &str) -> Result<Self> {
        let mut v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json).unwrap();

        let model = v.remove("model").unwrap().as_str().unwrap().to_string();
        let llm_service_rate_limit = v
            .remove("llm_service_rate_limit")
            .unwrap()
            .as_u64()
            .unwrap();
        let extra = serde_json::to_string(&v).unwrap();

        Ok(Self {
            model,
            llm_service_rate_limit,
            extra,
        })
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        tracing::trace!("Writing ScrapperConfig to the database {:?}", self);
        conn.execute(
            "INSERT INTO scrapper_config (bookkeeper_lnk, model, llm_service_rate_limit, extra) VALUES (?, ?, ?, ?)",
            [
                commit_id.to_string(),
                self.model.to_string(),
                self.llm_service_rate_limit.to_string(),
                self.extra.to_string(),
            ],
        )?;

        Ok(())
    }
}

#[cfg(test)]
/// Samples of scrapper configuration to support tests
///
/// These samples should cover multiple versions of data models as this library evolves and it
/// should be acessible from other parts of the crate.
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    pub(crate) fn as_text_v1() -> String {
        r#"
    {
      "log_level": "DEBUG",
      "out_dir": ".",
      "county_fp": "counties.csv",
      "model": "gpt-4",
      "llm_call_kwargs":{
        "temperature": 0,
        "seed": 42,
        "timeout": 300
        },
      "llm_service_rate_limit": 50000,
      "td_kwargs": {
        "dir": "."
        },
      "tpe_kwargs": {
        "max_workers": 10
        },
      "ppe_kwargs": {
        "max_workers": 4
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
mod test_scrapper_config {
    use super::sample::as_text_v1;
    use super::ScrapperConfig;

    #[test]
    /// Load a ScrapperConfig from a JSON string
    fn parse_json() {
        let config = ScrapperConfig::from_json(&as_text_v1()).unwrap();

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.llm_service_rate_limit, 50000);
    }
}
