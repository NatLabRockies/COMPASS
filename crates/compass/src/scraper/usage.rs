//! Parse and handle the Scrapper usage information

use std::io::Read;

use crate::error::Result;

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct UsageValues {
    //event: String,
    request: u32,
    prompt_tokens: u32,
    response_tokens: u32,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ScrapperUsage {
    pub(crate) total_time: f64,
    pub(crate) extra: String,
}

impl ScrapperUsage {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        tracing::trace!("Initializing database for ScrapperUsage");
        conn.execute_batch(
            r"
            CREATE SEQUENCE usage_sequence START 1;
            CREATE TABLE usage (
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              total_time FLOAT NOT NULL,
              extra TEXT,
              created_at TIMESTAMP NOT NULL DEFAULT NOW(),
              );

            CREATE SEQUENCE usage_per_item_sequence START 1;
            CREATE TABLE usage_per_item(
              id INTEGER PRIMARY KEY DEFAULT NEXTVAL('usage_per_item_sequence'),
              /* connection with file */
              jurisdiction_lnk INTEGER REFERENCES jurisdiction(id) NOT NULL,
              total_time FLOAT,
              total_requests INTEGER NOT NULL,
              total_prompt_tokens INTEGER NOT NULL,
              total_response_tokens INTEGER NOT NULL,
              );

            CREATE SEQUENCE usage_event_sequence START 1;
            CREATE TABLE usage_event (
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

    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Self> {
        tracing::trace!("Opening ScrapperUsage from {:?}", root.as_ref());

        let path = root.as_ref().join("usage.json");
        if !path.exists() {
            tracing::error!("Missing usage file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing usage file".to_string(),
            ));
        }

        tracing::trace!("Identified ScrapperUsage at {:?}", path);

        let file = std::fs::File::open(path);
        let mut reader = std::io::BufReader::new(file.unwrap());
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer);

        let usage = Self::from_json(&buffer)?;
        tracing::trace!("ScrapperUsage loaded: {:?}", usage);

        Ok(usage)
    }

    #[allow(dead_code)]
    pub(super) fn from_json(json: &str) -> Result<Self> {
        let mut v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json).unwrap();

        let total_time = v.remove("total_time_seconds").unwrap().as_f64().unwrap();
        let extra = serde_json::to_string(&v).unwrap();

        Ok(Self { total_time, extra })
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        tracing::trace!("Writing ScrapperUsage to the database {:?}", self);
        conn.execute(
            "INSERT INTO usage (total_time, extra) VALUES (?, ?, ?)",
            [
                &commit_id.to_string(),
                &self.total_time.to_string(),
                &self.extra,
            ],
        )?;
        tracing::trace!("ScrapperUsage written to the database");

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
          "total_time_seconds": 294.69257712364197,
          "total_time": "0:04:54.692577",
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
            "total_time_seconds": 294.64539074897766,
            "total_time": "0:04:54.645391",
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
        let usage = super::ScrapperUsage::from_json(&as_text_v1()).unwrap();

        assert!((usage.total_time - 294.69257712364197).abs() <= f64::EPSILON);
    }
}
