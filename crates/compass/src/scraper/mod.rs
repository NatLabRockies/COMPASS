//! Support for the ordinance scrapper output

mod metadata;
mod ordinance;
mod source;
mod usage;

use std::path::{Path, PathBuf};

use tracing::{self, trace};

use crate::error;
use crate::error::Result;
use metadata::Metadata;
use ordinance::Ordinance;
#[allow(unused_imports)]
use source::Source;
use usage::Usage;

pub(crate) const SCRAPPED_ORDINANCE_VERSION: &str = "0.0.1";

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

// Some concepts:
//
// - One single ordinance output is loaded and abstracted as a
//   ScrappedOrdinance. Everything inside should be accessible through this
//   abstraction.
// - It is possible to operate in multiple ordinance outputs at once, such
//   as loading multiple ordinance outputs into the database.
// - The ScrappedOrdinance should implement a hash estimate, which will
//   be used to identify the commit in the database.
// - Open ScrappedOrdinance is an async operation, and accessing/parsing
//   each component is also async. Thus, it can load into DB as it goes
//   until complete all components.
// - The sequence:
//   - Open ScrappedOrdinance (async)
//   - Validate the content (async)
//     - Does it has all the requirements?
//     - Light check. Without requiring to open everything or loading
//       everything in memory, does it look fine?
//   - Load into DB as each component is available (async)

#[allow(dead_code)]
#[derive(Debug)]
/// Abstraction for the ordinance scrapper raw output
///
/// The ordinance scrapper outputs a standard directory with multiple files
/// and sub-directories. This struct abstracts the access to such output.
pub(crate) struct ScrappedOrdinance {
    format_version: String,
    root: PathBuf,
    metadata: Metadata,
    sources: Vec<source::Source>,
    usage: Usage,
    ordinance: Ordinance,
}

impl ScrappedOrdinance {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        tracing::trace!("Initializing ScrappedOrdinance database");
        metadata::Metadata::init_db(conn)?;
        source::Source::init_db(conn)?;
        usage::Usage::init_db(conn)?;
        ordinance::Ordinance::init_db(conn)?;

        Ok(())
    }

    // Keep in mind a lazy state.
    #[allow(dead_code)]
    /// Open an existing scrapped ordinance folder
    pub(crate) async fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        trace!("Opening scrapped ordinance");

        let root = root.as_ref().to_path_buf();
        trace!("Scrapper output located at: {:?}", root);

        // Do some validation before returning a ScrappedOrdinance

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
        */
        let sources = source::Source::open(&root).await?;
        let metadata = metadata::Metadata::open(&root).await?;
        let usage = usage::Usage::open(&root).await?;
        let ordinance = ordinance::Ordinance::open(&root).await?;

        trace!("Scrapped ordinance opened successfully");
        Ok(Self {
            root,
            format_version: SCRAPPED_ORDINANCE_VERSION.to_string(),
            metadata,
            sources,
            usage,
            ordinance,
        })
    }

    #[allow(dead_code)]
    pub(crate) async fn push(&self, conn: &mut duckdb::Connection, commit_id: usize) -> Result<()> {
        // Load the ordinance into the database
        tracing::trace!("Pushing scrapped ordinance into the database");
        let conn = conn.transaction().unwrap();
        tracing::trace!("Transaction started");

        // Do I need to extract the hash here from the full ScrappedOutput?
        // What about username?
        self.sources
            .iter()
            .for_each(|s| s.write(&conn, commit_id).unwrap());
        self.metadata.write(&conn, commit_id).unwrap();
        self.usage().await.unwrap().write(&conn, commit_id).unwrap();
        self.ordinance.write(&conn, commit_id).unwrap();
        // commit transaction

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
    use super::ScrappedOrdinance;
    use super::metadata;
    use super::ordinance;
    use super::usage;
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

        let ordinance_files_path = target.path().join("ordinance_files");
        std::fs::create_dir(&ordinance_files_path).unwrap();
        let source_filename = ordinance_files_path.join("source.pdf");
        let mut source_file = std::fs::File::create(source_filename).unwrap();
        writeln!(source_file, "This is a sample ordinance file").unwrap();

        let _metadata_file = metadata::sample::as_file(target.path().join("meta.json")).unwrap();
        let _usage_file = usage::sample::as_file(target.path().join("usage.json")).unwrap();
        let _ordinance_file =
            ordinance::sample::as_file(target.path().join("quantitative_ordinances.csv")).unwrap();

        let demo = ScrappedOrdinance::open(target).await.unwrap();
        dbg!(&demo);

        /*
         * Just for reference. It now breaks the new design
        let tmp = tempfile::tempdir().unwrap();
        let db_filename = tmp.path().join("test.db");
        crate::init_db(db_filename.as_path().to_str().unwrap()).unwrap();

        // let mut db = duckdb::Connection::open_in_memory().unwrap();
        let mut db = duckdb::Connection::open(db_filename).unwrap();
        let conn = db.transaction().unwrap();
        ScrappedOrdinance::init_db(&conn).unwrap();
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
