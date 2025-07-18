//! Support for the ordinance scraper output

mod metadata;
mod ordinance;
mod source;
mod usage;

use std::path::{Path, PathBuf};

use tracing::{self, debug, trace};

use crate::error;
use crate::error::Result;
use metadata::Metadata;
use ordinance::Ordinance;
#[allow(unused_imports)]
use source::Source;
use usage::Usage;

pub(crate) const SCRAPED_ORDINANCE_VERSION: &str = "0.0.1";

// An arbitrary limit (5MB) to protect against maliciously large JSON files
const MAX_JSON_FILE_SIZE: u64 = 5 * 1024 * 1024;

// Concepts
// - Lazy loading a scraper output
//   - Early validation. Not necessary complete, but able to abort early
//     if identifies any major problem.
//   - Handle multiple versions. Identify right the way if the output is
//     a compatible version, and how to handle it.
//     - Define the trait and implement that on multiple readers for different
//       versions.
// - Async loading into DB
// - Logging operations

// Some concepts:
//
// - One single ordinance output is loaded and abstracted as a
//   ScrapedOrdinance. Everything inside should be accessible through this
//   abstraction.
// - It is possible to operate in multiple ordinance outputs at once, such
//   as loading multiple ordinance outputs into the database.
// - The ScrapedOrdinance should implement a hash estimate, which will
//   be used to identify the commit in the database.
// - Open ScrapedOrdinance is an async operation, and accessing/parsing
//   each component is also async. Thus, it can load into DB as it goes
//   until complete all components.
// - The sequence:
//   - Open ScrapedOrdinance (async)
//   - Validate the content (async)
//     - Does it has all the requirements?
//     - Light check. Without requiring to open everything or loading
//       everything in memory, does it look fine?
//   - Load into DB as each component is available (async)

#[allow(dead_code)]
#[derive(Debug)]
/// Abstraction for the ordinance scraper raw output
///
/// The ordinance scraper outputs a directory with a standard structure,
/// including multiple files and sub-directories. The `ScrapedOrdinance`
/// compose all that information.
pub(crate) struct ScrapedOrdinance {
    /// The data model version
    format_version: String,
    /// The root path of the scraped ordinance output
    root: PathBuf,
    /// The metadata section
    metadata: Metadata,
    /// The source section
    source: Source,
    /// The usage section
    usage: Usage,
    /// The ordinance section
    ordinance: Ordinance,
}

impl ScrapedOrdinance {
    /// Initialize the database schema for the scraped ordinance
    ///
    /// This function creates the necessary tables and resources
    /// in the database to store the scraped ordinance data model
    /// by calling each component, in the correct order.
    ///
    /// # Arguments
    ///
    /// `conn`: A reference to the database transaction
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        debug!("Initializing ScrapedOrdinance database");

        source::Source::init_db(conn)?;
        metadata::Metadata::init_db(conn)?;
        usage::Usage::init_db(conn)?;
        ordinance::Ordinance::init_db(conn)?;

        Ok(())
    }

    // Keep in mind a lazy state.
    #[allow(dead_code)]
    /// Open an existing scraped ordinance folder
    pub(crate) async fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        trace!("Opening scraped ordinance");

        let root = root.as_ref().to_path_buf();
        trace!("Scraper output located at: {:?}", root);

        // Do some validation before returning a ScrapedOrdinance

        if !root.exists() {
            trace!("Root path does not exist: {:?}", root);
            return Err(error::Error::Undefined("Path does not exist".to_string()));
        }

        let (source, metadata, usage, ordinance) = tokio::try_join!(
            source::Source::open(&root),
            metadata::Metadata::open(&root),
            usage::Usage::open(&root),
            ordinance::Ordinance::open(&root)
        )?;
        trace!("Scraped ordinance opened successfully");

        Ok(Self {
            root,
            format_version: SCRAPED_ORDINANCE_VERSION.to_string(),
            metadata,
            source,
            usage,
            ordinance,
        })
    }

    #[allow(dead_code)]
    pub(crate) async fn push(&self, conn: &mut duckdb::Connection, commit_id: usize) -> Result<()> {
        // Load the ordinance into the database
        tracing::trace!("Pushing scraped ordinance into the database");
        let conn = conn.transaction().unwrap();
        tracing::trace!("Transaction started");

        // Do I need to extract the hash here from the full ScrapedOutput?
        // What about username?
        self.source.record(&conn, commit_id).unwrap();
        self.metadata.write(&conn, commit_id).unwrap();
        self.usage().await.unwrap().write(&conn, commit_id).unwrap();
        self.ordinance.write(&conn, commit_id).unwrap();

        tracing::trace!("Committing transaction");
        conn.commit()?;

        Ok(())
    }

    #[allow(dead_code)]
    async fn usage(&self) -> Result<Usage> {
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
    use super::ScrapedOrdinance;
    use super::metadata;
    use super::ordinance;
    use super::source;
    use super::usage;
    use std::io::Write;

    #[tokio::test]
    /// Opening an inexistent path should give an error
    async fn open_inexistent_path() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("inexistent");

        // First confirm that the path does not exist
        assert!(!target.exists());
        ScrapedOrdinance::open(target).await.unwrap_err();
    }

    #[tokio::test]
    /// Open a Scraped Ordinance raw output
    async fn open_scraped_ordinance() {
        // A sample ordinance file for now.
        let target = tempfile::tempdir().unwrap();

        let _source_file =
            source::sample::as_file(target.path().join("jurisdictions.json")).unwrap();
        let ordinance_files_path = target.path().join("ordinance_files");
        std::fs::create_dir(&ordinance_files_path).unwrap();
        let source_filename = ordinance_files_path.join("source.pdf");
        let mut source_file = std::fs::File::create(source_filename).unwrap();
        writeln!(source_file, "This is a sample ordinance file").unwrap();

        let _metadata_file = metadata::sample::as_file(target.path().join("meta.json")).unwrap();
        let _usage_file = usage::sample::as_file(target.path().join("usage.json")).unwrap();
        ordinance::sample::as_file(target.path()).unwrap();

        let demo = ScrapedOrdinance::open(target).await.unwrap();
        dbg!(&demo);

        /*
         * Just for reference. It now breaks the new design
        let tmp = tempfile::tempdir().unwrap();
        let db_filename = tmp.path().join("test.db");
        crate::init_db(db_filename.as_path().to_str().unwrap()).unwrap();

        // let mut db = duckdb::Connection::open_in_memory().unwrap();
        let mut db = duckdb::Connection::open(db_filename).unwrap();
        let conn = db.transaction().unwrap();
        ScrapedOrdinance::init_db(&conn).unwrap();
        let username = "test";
        let commit_id: usize = conn.query_row(
            "INSERT INTO bookkeeper (hash, username) VALUES (?, ?) RETURNING id",
            ["dummy hash".to_string(), username.to_string()],
            |row| row.get(0),
            ).expect("Failed to insert into bookkeeper");
        conn.commit().unwrap();
        demo.push(&mut db, commit_id).await.unwrap();
        */
    }
}
