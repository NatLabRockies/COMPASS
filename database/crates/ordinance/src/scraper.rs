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
/// Abstraction for the ordinance scrapper raw output
///
/// The ordinance scrapper outputs a standard directory with multiple files
/// and sub-directories. This struct abstracts the access to such output.
struct ScrappedOrdinance {
    format_version: String,
    root: PathBuf,
}

impl ScrappedOrdinance {
    // Keep in mind a lazy state.
    #[allow(dead_code)]
    /// Open an existing scrapped ordinance folder
    async fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
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
    fn config(&self) -> Result<serde_json::Value> {
        let config_file = &self.root.join("config.json");
        if !config_file.exists() {
            trace!("Missing config file: {:?}", config_file);
            return Err(error::Error::Undefined(
                "Features file does not exist".to_string(),
            ));
        }

        let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(config_file)?)
            .expect("Failed to parse config file");
        dbg!(&config);

        Ok(config)
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
