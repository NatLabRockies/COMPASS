#![deny(missing_docs)]

//! NREL's ordinance database

mod error;
mod scraper;

use duckdb::Connection;
use serde::Serialize;
use tracing::{self, trace};

use error::Result;

/// Initialize the database
///
/// Create a new database as a local single file ready to store the ordinance
/// data.
pub fn init_db(path: &str) -> Result<()> {
    trace!("Creating a new database at {:?}", &path);

    let mut db = Connection::open(path)?;
    trace!("Database opened: {:?}", &db);

    db.execute_batch("SET VARIABLE ordinancedb_version = '0.0.1';")?;
    trace!("Defining ordinance data model version as: 0.0.1");

    /*
     * Change the source structure to have a database of all sources, and
     * the run links to the source used. Also link that to the jurisdiction
     * such that it could later request all sources related to a certain
     * jurisdiction.
     *
     * Multiple sources for the same jurisdiction (consider multiple
     * technologies) should be possible.
     *
     * In case of multiple sources for one jurisdiction, we should be able
     * to support what was the latest document available, or list all of
     * them.
     *
     * Check the new jurisdiction file.
     *
     * If user don't have a database. Download it in the right path.
     *
     *
     */
    trace!("Creating table bookkeeper");
    db.execute_batch(
        "BEGIN;
    CREATE SEQUENCE bookkeeper_sequence START 1;
    CREATE TABLE bookkeeper (
        id INTEGER PRIMARY KEY DEFAULT NEXTVAL('bookkeeper_sequence'),
        hash TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        username TEXT,
        comment TEXT,
        model TEXT
        );

    INSTALL spatial;
    LOAD spatial;

    CREATE SEQUENCE jurisdiction_sequence START 1;
    CREATE TYPE jurisdiction_rank AS ENUM ('state', 'county', 'parish', 'city', 'town', 'district', 'other');
    CREATE TABLE jurisdiction (
      id INTEGER PRIMARY KEY DEFAULT NEXTVAL('jurisdiction_sequence'),
      bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
      name TEXT NOT NULL,
      FIPS UBIGINT NOT NULL,
      geometry GEOMETRY NOT NULL,
      rank jurisdiction_rank NOT NULL,
      parent_id INTEGER REFERENCES jurisdiction(id),
      created_at TIMESTAMP NOT NULL DEFAULT NOW(),
      src TEXT,
      comments TEXT
      );

    COMMIT;",
    )?;

    let conn = db.transaction()?;
    scraper::ScrapedOrdinance::init_db(&conn)?;
    conn.commit()?;

    println!("{}", db.is_autocommit());

    trace!("Database initialized");
    Ok(())
}

/// Scan and load features from a CSV file
///
/// Proof of concept. Parse a CSV file and load the features into the
/// database.
pub fn load_ordinance<P: AsRef<std::path::Path> + std::fmt::Debug>(
    mut database: duckdb::Connection,
    username: &String,
    ordinance_path: P,
) -> Result<()> {
    // insert into bookkeeper (hash, username) and get the pk to be used in all the following
    // inserts.
    trace!("Starting a transaction");
    let conn = database.transaction().unwrap();

    let commit_id: usize = conn
        .query_row(
            "INSERT INTO bookkeeper (hash, username) VALUES (?, ?) RETURNING id",
            ["dummy hash".to_string(), username.to_string()],
            |row| row.get(0),
        )
        .expect("Failed to insert into bookkeeper");

    tracing::debug!("Commit id: {:?}", commit_id);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let ordinance = runtime
        .block_on(scraper::ScrapedOrdinance::open(ordinance_path))
        .unwrap();
    conn.commit().unwrap();
    tracing::debug!("Transaction committed");

    trace!("Ordinance: {:?}", ordinance);
    runtime
        .block_on(ordinance.push(&mut database, commit_id))
        .unwrap();

    /*
    let mut rdr = csv::Reader::from_path(raw_filename).unwrap();
    let mut stmt = conn.prepare_cached("INSERT INTO property (county, state, FIPS, feature, fixed_value, mult_value, mult_type, adder, min_dist, max_dist, value, units, ord_year, last_updated, section, source, comments) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").unwrap();
    for result in rdr.records() {
        let record = result.unwrap();
        // println!("{:?}", record);
        stmt.execute([
            record[0].to_string(),
            record[1].to_string(),
            record[2].to_string(),
            record[3].to_string(),
            record[4].to_string(),
            record[5].to_string(),
            record[6].to_string(),
            record[7].to_string(),
            record[8].to_string(),
            record[9].to_string(),
            record[10].to_string(),
            record[11].to_string(),
            record[12].to_string(),
            record[13].to_string(),
            record[14].to_string(),
            record[15].to_string(),
            record[16].to_string(),
        ])
        .unwrap();
    }

    */
    //let df = polars::io::csv::read::CsvReadOptions::default().with_has_header(true).try_into_reader_with_file_path(Some("sample.csv".into())).unwrap().finish();

    Ok(())
}

#[allow(dead_code, non_snake_case)]
#[derive(Debug, Serialize)]
/// Ordinance record that combines quantitative and qualitative
///
/// It is currently used for handle reVX ordinance for reVX standard only
/// but we might expand this in the future to handle more complete outputs.
struct OrdinanceRecord {
    /// FIPS code of the jurisdiction
    FIPS: u64,
    /// Feature type, e.g., "setback", "height", etc.
    feature: String,
    /// Feature subtype, currently not used but required by reVX standard.
    feature_subtype: Option<String>,
    /// Quantitative feature value, e.g., 3.14
    quantitative: Option<f64>,
    /// Qualitative feature value, e.g., "color of the tips of the blades"
    qualitative: Option<String>,
}

#[derive(Debug)]
/// Technologies supported by the ordinance database
enum Technology {
    /// Everything related to wind energy
    Wind,
    /// Everything related to solar energy
    Solar,
}

impl std::fmt::Display for Technology {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Technology::Wind => write!(f, "wind"),
            Technology::Solar => write!(f, "solar"),
        }
    }
}

impl std::convert::TryFrom<&str> for Technology {
    type Error = error::Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            "wind" => Ok(Technology::Wind),
            "solar" => Ok(Technology::Solar),
            _ => Err(error::Error::Undefined(format!("Unknown technology: {s}"))),
        }
    }
}

/// Export the database
///
/// Currently, it is a proof of concept. It reads the database and prints
/// some fields to the standard output in CSV format.
pub fn export<W: std::io::Write>(
    wtr: &mut W,
    db_filename: &str,
    format: &str,
    technology: &str,
) -> Result<()> {
    trace!("Exporting database: {:?}", db_filename);
    // Not used yet, but ready for future use
    trace!("Export format: {:?}", format);

    let technology = Technology::try_from(technology)?;

    let conn = Connection::open(db_filename)?;
    trace!("Database opened: {:?}", &conn);

    let mut stmt = conn
        .prepare( &format!("SELECT FIPS, feature, feature_subtype, quantitative, qualitative FROM ordinance JOIN scraper_metadata ON (ordinance.bookkeeper_lnk=scraper_metadata.bookkeeper_lnk) WHERE scraper_metadata.technology='{technology}' ORDER BY FIPS, feature;")
            )
        .expect("Failed to prepare statement");
    //dbg!("Row count", stmt.row_count());
    let row_iter = stmt
        .query_map([], |row| {
            Ok(OrdinanceRecord {
                FIPS: row.get(0)?,
                feature: row.get(1)?,
                feature_subtype: row.get(2)?,
                quantitative: row.get(3)?,
                qualitative: row.get(4)?,
            })
        })
        .expect("Failed to query");

    let mut wtr = csv::Writer::from_writer(wtr);

    for row in row_iter {
        wtr.serialize(row?)?;
    }
    wtr.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let _ = init_db("test");
    }
}
