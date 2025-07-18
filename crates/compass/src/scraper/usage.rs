//! All the context for the scraper usage data
//!
//! This module provides support to parse, define the required database
//! structure and record data in the database. All the context specific
//! for the scraper usage is defined here.

use std::collections::HashMap;
use std::io::Read;

use tracing::debug;

use crate::error::Result;

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Scrapper usage data
///
/// This top level structure contains all the usage information for a single
/// run of the scraper. Given one run can contain multiple models, each
/// with multiple steps, each step having it's own usage information.
pub(super) struct Usage {
    #[serde(flatten)]
    pub(super) jurisdiction: HashMap<String, UsagePerModel>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Scraper usage for a single model
///
/// Holds the usage information for a single LLM of a single run of the
/// scraper. Each item is a hash map that tracks the usage per target
/// such as 'data extraction' or 'document validation'. All the
/// components are stored in the `model` field.
pub(super) struct UsagePerModel {
    #[serde(flatten)]
    model: HashMap<String, UsagePerStep>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Scraper usage for a single step
///
/// Holds the usage information for a single target of a single run of the
/// scraper. Each item has the totals as well as the information for specific
/// components such as 'data extraction' or 'document validation'. All the
/// components are stored in the `step` field.
pub(super) struct UsagePerStep {
    #[serde(flatten)]
    step: HashMap<String, UsageValues>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub(super) struct UsageValues {
    requests: u32,
    prompt_tokens: u32,
    response_tokens: u32,

    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

impl Usage {
    /// Initialize the database for the Usage context
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        tracing::trace!("Initializing database for Usage");
        conn.execute_batch(
            r"
            CREATE SEQUENCE usage_sequence START 1;
            CREATE TABLE IF NOT EXISTS usage (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              jurisdiction TEXT NOT NULL,
              );

            CREATE SEQUENCE usage_model_sequence START 1;
            CREATE TABLE IF NOT EXISTS usage_model(
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_model_sequence'),
              usage_lnk INTEGER REFERENCES usage(id) NOT NULL,
              model TEXT NOT NULL,
              total_requests INTEGER NOT NULL,
              total_prompt_tokens INTEGER NOT NULL,
              total_response_tokens INTEGER NOT NULL,
              );

            CREATE SEQUENCE usage_step_sequence START 1;
            CREATE TABLE IF NOT EXISTS usage_step (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_step_sequence'),
              model_lnk INTEGER REFERENCES usage_model(id) NOT NULL,
              step TEXT NOT NULL,
              requests INTEGER NOT NULL,
              prompt_tokens INTEGER NOT NULL,
              response_tokens INTEGER NOT NULL,
              );",
        )?;

        Ok(())
    }

    /// Open the usage related components of the scraper output
    ///
    /// # Arguments
    /// * `root`: The root directory of the scraper output.
    ///
    /// # Returns
    /// A Usage structure with the parsed data.
    ///
    /// # Attention
    /// Currently opens and parses right the way the usage data. In the future
    /// this should be changed to a lazy approach and take better advantage of
    /// been async.
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        debug!("Opening Usage from {:?}", root.as_ref());

        let path = root.as_ref().join("usage.json");
        if !path.exists() {
            tracing::error!("Missing usage file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing usage file".to_string(),
            ));
        }

        tracing::trace!("Identified Usage at {:?}", path);

        let file = std::fs::File::open(path);
        let mut reader = std::io::BufReader::new(file.unwrap());
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer);

        let usage = Self::from_json(&buffer)?;
        tracing::trace!("Usage loaded: {:?}", usage);

        Ok(usage)
    }

    /// Parse the usage data from a JSON string
    pub(super) fn from_json(json: &str) -> Result<Self> {
        tracing::trace!("Parsing Usage as JSON");
        let usage: Usage = serde_json::from_str(json).unwrap();
        Ok(usage)
    }

    /// Write the usage data to the database
    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        tracing::trace!("Writing Usage to the database {:?}", self);

        for (jurisdiction_name, usage_by_model) in &self.jurisdiction {
            tracing::trace!("Writing usage for {:?} to the database", jurisdiction_name);

            // An integer type in duckdb is 32 bits.
            let jurisdiction_id: u32 = conn
                .query_row(
                    "INSERT INTO usage (bookkeeper_lnk, jurisdiction) VALUES (?, ?) RETURNING id",
                    [&commit_id.to_string(), jurisdiction_name],
                    |row| row.get(0),
                )
                .expect("Failed to insert usage");

            tracing::trace!(
                "Usage per jurisdiction written to the database, id: {:?}",
                jurisdiction_id
            );

            for (model_name, content) in usage_by_model
                .model
                .iter()
                .filter(|(k, _)| *k != "tracker_totals")
            {
                tracing::trace!("Writing usage for model {:?} to the database", model_name);

                let model_id: u32 = conn.query_row(
                    "INSERT INTO usage_model (usage_lnk, model, total_requests, total_prompt_tokens, total_response_tokens) VALUES (?, ?, ?, ?, ?) RETURNING id",
                    [
                        &jurisdiction_id.to_string(),
                        model_name,
                        &usage_by_model.model["tracker_totals"].step[model_name].requests.to_string(),
                        &usage_by_model.model["tracker_totals"].step[model_name].prompt_tokens.to_string(),
                        &usage_by_model.model["tracker_totals"].step[model_name].response_tokens.to_string()
                    ],
                    |row| row.get(0)
                    ).expect("Failed to insert usage_per_jurisdiction");

                tracing::trace!(
                    "Usage per model written to the database, id: {:?}",
                    model_id
                );

                for (step_name, step) in &content.step {
                    tracing::trace!("Writing usage for step {:?} to the database", step_name);

                    conn.execute(
                        "INSERT INTO usage_step (model_lnk, step, requests, prompt_tokens, response_tokens) VALUES (?, ?, ?, ?, ?)",
                        [
                            &model_id.to_string(),
                            step_name,
                            &step.requests.to_string(),
                            &step.prompt_tokens.to_string(),
                            &step.response_tokens.to_string()
                        ]
                        ).expect("Failed to insert usage_step");
                }

                tracing::trace!("Usage per step written to the database");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    pub(crate) fn as_text_v1() -> String {
        r#"
        {
          "Decatur County, Indiana": {
            "gpt-4.1-mini": {
              "document_location_validation": {
                "requests": 55,
                "prompt_tokens": 114614,
                "response_tokens": 1262
              },
              "document_content_validation": {
                "requests": 7,
                "prompt_tokens": 15191,
                "response_tokens": 477
              }
            },
            "tracker_totals": {
              "gpt-4.1-mini": {
                "requests": 121,
                "prompt_tokens": 186099,
                "response_tokens": 6297
              }
            }
          }
        }"#
        .to_string()
    }

    pub(crate) fn as_file<P: AsRef<std::path::Path>>(path: P) -> Result<std::fs::File> {
        let mut f = std::fs::File::create(path)?;
        write!(f, "{}", as_text_v1()).unwrap();
        Ok(f)
    }
}

#[cfg(test)]
mod test_scraper_usage {
    use super::sample::as_text_v1;

    #[test]
    fn parse_json() {
        let usage = super::Usage::from_json(&as_text_v1()).unwrap();

        assert!(usage.jurisdiction.contains_key("Decatur County, Indiana"));
        assert!(
            usage.jurisdiction["Decatur County, Indiana"].model["gpt-4.1-mini"]
                .step
                .contains_key("document_location_validation")
        );
        assert_eq!(
            usage.jurisdiction["Decatur County, Indiana"].model["gpt-4.1-mini"]
                .step
                .get("document_location_validation")
                .unwrap()
                .requests,
            55
        );
    }
}
