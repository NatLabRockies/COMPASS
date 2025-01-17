//! Support for the ordinance scrapper output

use std::path::{Path, PathBuf};

use tracing::trace;

use crate::error;
use crate::error::Result;

// Concepts
// - Lazy loading a scrapper output
//   - Early validation. Not necessary complete, but able to abort early
//     if identifies any major problem.
//   - Handle multiple versions. Identify right the way if the output is
//     a compatible version, and how to handle it.
//     - Define the trait and implement that on multiple readers for different
//       versions.
// - Async loading into DB
// - Logging operations

#[allow(dead_code)]
#[derive(Debug)]
/// Configuration of the ordinance scrapper
pub(crate) struct ScrapperConfig {
    pub(crate) model: String,
    pub(crate) llm_service_rate_limit: u64,
    pub(crate) extra: String,
}

impl ScrapperConfig {
    #[allow(dead_code)]
    /// Extract the configuration from a JSON string
    fn from_json(json: &str) -> Result<Self> {
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
}

mod test_scrapper_config {
    use super::ScrapperConfig;

    #[test]
    fn dev() {
        let data = r#"
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
        }"#;

        let config = ScrapperConfig::from_json(data).unwrap();

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.llm_service_rate_limit, 50000);
    }
}

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
pub(crate) struct Usage {
    pub(crate) total_time: f64,
    pub(crate) extra: String,
}

impl Usage {
    #[allow(dead_code)]
    fn from_json(json: &str) -> Result<Self> {
        let mut v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json).unwrap();

        let total_time = v.remove("total_time_seconds").unwrap().as_f64().unwrap();
        let extra = serde_json::to_string(&v).unwrap();

        Ok(Self { total_time, extra })
    }
}

mod test_usage {
    #[test]
    fn dev() {
        let data = r#"
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
        }"#;
        let usage = super::Usage::from_json(data).unwrap();

        assert!((usage.total_time - 294.69257712364197).abs() <= f64::EPSILON);
    }
}

#[allow(dead_code)]
#[derive(Debug)]
/// Abstraction for the ordinance scrapper raw output
///
/// The ordinance scrapper outputs a standard directory with multiple files
/// and sub-directories. This struct abstracts the access to such output.
pub(crate) struct ScrappedOrdinance {
    format_version: String,
    root: PathBuf,
}

impl ScrappedOrdinance {
    // Keep in mind a lazy state.
    #[allow(dead_code)]
    /// Open an existing scrapped ordinance folder
    pub(crate) async fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        trace!("Opening scrapped ordinance");

        let root = root.as_ref().to_path_buf();
        trace!("Defined root as: {:?}", root);

        // Validate
        if !root.exists() {
            trace!("Root path does not exist");
            return Err(error::Error::Undefined("Path does not exist".to_string()));
        }

        /*
        let features_file = root.join("ord_db.csv");
        if !features_file.exists() {
            trace!("Missing features file: {:?}", features_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }

        let config_file = root.join("ord_db.csv");
        if !config_file.exists() {
            trace!("Missing config file: {:?}", config_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }

        let usage_file = root.join("ord_db.csv");
        if !usage_file.exists() {
            trace!("Missing usage file: {:?}", usage_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }
        */

        Ok(Self {
            root,
            format_version: "0.0.1".to_string(),
        })
    }

    #[allow(dead_code)]
    pub(crate) async fn config(&self) -> Result<ScrapperConfig> {
        let config_file = &self.root.join("config.json");
        if !config_file.exists() {
            trace!("Missing config file: {:?}", config_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }

        let config = ScrapperConfig::from_json(&std::fs::read_to_string(config_file)?)
            .expect("Failed to parse config file");

        Ok(config)
    }

    #[allow(dead_code)]
    pub(crate) async fn usage(&self) -> Result<Usage> {
        let usage_file = &self.root.join("usage.json");
        if !usage_file.exists() {
            trace!("Missing usage file: {:?}", usage_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }

        let usage = Usage::from_json(&std::fs::read_to_string(usage_file)?)
            .expect("Failed to parse usage file");

        Ok(usage)
    }
}

#[cfg(test)]
mod tests {
    use super::ScrappedOrdinance;
    use std::io::Write;

    #[tokio::test]
    /// Opening an inexistent path should give an error
    async fn open_inexistent_path() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("inexistent");

        // First confirm that the path does not exist
        assert!(!target.exists());

        ScrappedOrdinance::open(target).await.unwrap_err();
    }

    #[tokio::test]
    /// Open a Scrapped Ordinance raw output
    async fn open_scrapped_ordinance() {
        // A sample ordinance file for now.
        let target = tempfile::tempdir().unwrap();
        let ordinance_filename = target.path().join("ord_db.csv");
        let mut ordinance_file = std::fs::File::create(ordinance_filename).unwrap();
        writeln!(ordinance_file, "county,state,FIPS,feature,fixed_value,mult_value,mult_type,adder,min_dist,max_dist,value,units,ord_year,last_updated,section,source,comment").unwrap();
        writeln!(ordinance_file, "county-1,state-1,FIPS-1,feature-1,fixed_value-1,mult_value-1,mult_type-1,adder-1,min_dist-1,max_dist-1,value-1,units-1,ord_year-1,last_updated-1,section-1,source-1,comment").unwrap();

        ScrappedOrdinance::open(target).await.unwrap();
    }
}
