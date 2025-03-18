//! All the context for the scrapper usage data
//!
//! This module provides support to parse, define the required database
//! structure and record data in the database. All the context specific
//! for the scrapper usage is defined here.

use std::collections::HashMap;
use std::io::Read;

use crate::error::Result;

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Scrapper usage data
///
/// This top level structure contains all the usage information for a single
/// run of the scrapper. Given one run can contain multiple targets, each
/// target is one item in the jurisdiction item.
pub(super) struct Usage {
    // pub(super) total_time_seconds: f64,
    // pub(super) total_time: String,
    #[serde(flatten)]
    pub(super) jurisdiction: HashMap<String, UsagePerItem>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
/// Scraper usage for a single target
///
/// Holds the usage information for a single target of a single run of the
/// scrapper. Each item has the totals as well as the information for specific
/// components such as 'data extraction' or 'document validation'. All the
/// components are stored in the `events` field.
pub(super) struct UsagePerItem {
    // total_time_seconds: f64,
    // total_time: String,
    #[serde(flatten)]
    events: HashMap<String, UsageValues>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub(super) struct UsageValues {
    //event: String,
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
              created_at TIMESTAMP NOT NULL DEFAULT NOW(),
              );

            CREATE SEQUENCE usage_per_item_sequence START 1;
            CREATE TABLE IF NOT EXISTS usage_per_item(
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_per_item_sequence'),
              name TEXT NOT NULL,
              /* connection with file
              jurisdiction_lnk INTEGER REFERENCES jurisdiction(id) NOT NULL,
              */
              total_requests INTEGER NOT NULL,
              total_prompt_tokens INTEGER NOT NULL,
              total_response_tokens INTEGER NOT NULL,
              );

            CREATE SEQUENCE usage_event_sequence START 1;
            CREATE TABLE IF NOT EXISTS usage_event (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_event_sequence'),
              usage_per_item_lnk INTEGER REFERENCES usage_per_item(id) NOT NULL,
              event TEXT NOT NULL,
              requests INTEGER NOT NULL,
              prompt_tokens INTEGER NOT NULL,
              response_tokens INTEGER NOT NULL,
              );",
        )?;

        Ok(())
    }

    /// Open the usage related components of the scrapper output
    ///
    /// # Arguments
    /// * `root`: The root directory of the scrapper output.
    ///
    /// # Returns
    /// A Usage structure with the parsed data.
    ///
    /// # Attention
    /// Currently opens and parses right the way the usage data. In the future
    /// this should be changed to a lazy approach and take better advantage of
    /// been async.
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        tracing::trace!("Opening Usage from {:?}", root.as_ref());

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
        // An integer type in duckdb is 32 bits.
        let usage_id: u32 = conn
            .query_row(
                "INSERT INTO usage (bookkeeper_lnk) VALUES (?) RETURNING id",
                [&commit_id.to_string()],
                |row| row.get(0),
            )
            .expect("Failed to insert usage");
        tracing::trace!("Usage written to the database, id: {:?}", usage_id);

        for (jurisdiction_name, content) in &self.jurisdiction {
            tracing::trace!(
                "Writing Usage-Item to the database: {:?}",
                jurisdiction_name
            );

            // An integer type in duckdb is 32 bits.
            let item_id: u32 = conn.query_row(
                "INSERT INTO usage_per_item (name, total_requests, total_prompt_tokens, total_response_tokens) VALUES (?, ?, ?, ?) RETURNING id",
                [jurisdiction_name, &content.events["tracker_totals"].requests.to_string(), &content.events["tracker_totals"].prompt_tokens.to_string(), &content.events["tracker_totals"].response_tokens.to_string()],
                |row| row.get(0)
                ).expect("Failed to insert usage_per_item");

            tracing::trace!("UsagePerItem written to the database, id: {:?}", item_id);

            for (event_name, event) in &content.events {
                tracing::trace!("Writing Usage-Event to the database: {:?}", event_name);

                conn.execute(
                    "INSERT INTO usage_event (usage_per_item_lnk, event, requests, prompt_tokens, response_tokens) VALUES (?, ?, ?, ?, ?)",
                    [&item_id.to_string(), event_name, &event.requests.to_string(), &event.prompt_tokens.to_string(), &event.response_tokens.to_string()]
                    ).expect("Failed to insert usage_event");
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
            "document_location_validation": {
              "requests": 55,
              "prompt_tokens": 114614,
              "response_tokens": 1262
            },
            "document_content_validation": {
              "requests": 7,
              "prompt_tokens": 15191,
              "response_tokens": 477
            },
            "tracker_totals": {
              "requests": 121,
              "prompt_tokens": 186099,
              "response_tokens": 6297
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
mod test_scrapper_usage {
    use super::sample::as_text_v1;

    #[test]
    fn parse_json() {
        let usage = super::Usage::from_json(&as_text_v1()).unwrap();

        assert!(usage.jurisdiction.contains_key("Decatur County, Indiana"));
        assert!(
            usage.jurisdiction["Decatur County, Indiana"]
                .events
                .contains_key("document_location_validation")
        );
        assert_eq!(
            usage.jurisdiction["Decatur County, Indiana"]
                .events
                .get("document_location_validation")
                .unwrap()
                .requests,
            55
        );
    }
}
