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
    CREATE TYPE jurisdiction_rank AS ENUM ('state', 'county', 'city', 'town', 'CCD', 'reservation', 'other');
    CREATE TABLE jurisdiction (
      id INTEGER PRIMARY KEY DEFAULT NEXTVAL('jurisdiction_sequence'),
      bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
      name TEXT NOT NULL,
      FIPS INTEGER NOT NULL,
      geometry GEOMETRY NOT NULL,
      rank jurisdiction_rank NOT NULL,
      parent_id INTEGER REFERENCES jurisdiction(id),
      created_at TIMESTAMP NOT NULL DEFAULT NOW(),
      src TEXT,
      comments TEXT
      );

    CREATE SEQUENCE property_sequence START 1;
    CREATE TABLE property (
      id INTEGER PRIMARY KEY DEFAULT NEXTVAL('property_sequence'),

      county TEXT,
      state TEXT,
      FIPS INTEGER,
      feature TEXT NOT NULL,
      fixed_value TEXT,
      mult_value TEXT,
      mult_type TEXT,
      adder TEXT,
      min_dist TEXT,
      max_dist TEXT,
      value TEXT,
      units TEXT,
      ord_year TEXT,
      last_updated TEXT,
      section TEXT,
      source TEXT,

      comments TEXT
      );

    COMMIT;",
    )?;

    let conn = db.transaction()?;
    scraper::ScrappedOrdinance::init_db(&conn)?;
    conn.commit()?;

    println!("{}", db.is_autocommit());

    trace!("Database initialized");
    Ok(())
}

#[allow(dead_code)]
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
    tracing::trace!("Starting a transaction");
    let conn = database.transaction().unwrap();

    let commit_id: usize = conn
        .query_row(
            "INSERT INTO bookkeeper (hash, username) VALUES (?, ?) RETURNING id",
            ["dummy hash".to_string(), username.to_string()],
            |row| row.get(0),
        )
        .expect("Failed to insert into bookkeeper");

    tracing::debug!("Commit id: {:?}", commit_id);

    /*
    dbg!(&ordinance_path);
    let raw_filename = ordinance_path.as_ref().join("ord_db.csv");
    dbg!(&raw_filename);
    dbg!("========");
    */

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let ordinance = rt
        .block_on(scraper::ScrappedOrdinance::open(ordinance_path))
        .unwrap();
    conn.commit().unwrap();
    tracing::debug!("Transaction committed");

    tracing::trace!("Ordinance: {:?}", ordinance);
    rt.block_on(ordinance.push(&mut database, commit_id))
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

#[derive(Debug, Serialize)]
struct Ordinance {
    county: String,
    state: String,
    fips: i32,
    feature: String,
}

/// Export the database
///
/// Currently, it is a proof of concept. It reads the database and prints
/// some fields to the standard output in CSV format.
pub fn export_db(db_filename: &str) {
    let conn = Connection::open(db_filename).unwrap();
    let mut stmt = conn
        .prepare("SELECT county, state, fips, feature FROM property")
        .expect("Failed to prepare statement");
    //dbg!("Row count", stmt.row_count());
    let row_iter = stmt
        .query_map([], |row| {
            Ok(Ordinance {
                county: row.get(0)?,
                state: row.get(1)?,
                fips: row.get(2)?,
                feature: row.get(3)?,
            })
        })
        .expect("Failed to query");

    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    for row in row_iter {
        wtr.serialize(row.unwrap()).unwrap();
    }
    wtr.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let _ = init_db("test");
    }
}
