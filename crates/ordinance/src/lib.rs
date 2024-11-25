#![deny(missing_docs)]

//! NREL's ordinance database

mod error;

use duckdb::Connection;
use error::Result;

/// Initialize the database
///
/// Create a new database as a local single file ready to store the ordinance
/// data.
pub fn init_db(path: &str) -> Result<()> {
    let db = Connection::open(path)?;
    db.execute_batch("SET VARIABLE ordinancedb_version = '0.0.1';")?;

    db.execute_batch(
        "BEGIN;
    CREATE SEQUENCE bookkeeping_sequence START 1;
    CREATE TABLE bookkeeping (
        id INTEGER PRIMARY KEY DEFAULT NEXTVAL('bookkeeping_sequence'),
        hash TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        username TEXT,
        comment TEXT,
        model TEXT
        );

    CREATE SEQUENCE source_sequence START 1;
    CREATE TABLE source (
      id INTEGER PRIMARY KEY DEFAULT NEXTVAL('source_sequence'),
      bookkeeping_lnk INTEGER REFERENCES bookkeeping(id) NOT NULL,
      name TEXT NOT NULL,
      hash TEXT NOT NULL,
      origin TEXT,
      access_time TIMESTAMP,
      created_at TIMESTAMP NOT NULL DEFAULT NOW(),
      comments TEXT
      );

    INSTALL spatial;
    LOAD spatial;

    CREATE SEQUENCE jurisdiction_sequence START 1;
    CREATE TYPE jurisdiction_rank AS ENUM ('state', 'county', 'township', 'other');
    CREATE TABLE jurisdiction (
      id INTEGER PRIMARY KEY DEFAULT NEXTVAL('jurisdiction_sequence'),
      bookkeeping_lnk INTEGER REFERENCES bookkeeping(id) NOT NULL,
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

    println!("{}", db.is_autocommit());
    Ok(())
}

#[allow(dead_code)]
/// Scan and load features from a CSV file
///
/// Proof of concept. Parse a CSV file and load the features into the
/// database.
fn scan_features(db_filename: &str, raw_filename: &str) {
    let conn: Connection = Connection::open(db_filename).unwrap();

    let mut rdr = csv::Reader::from_path(raw_filename).unwrap();
    let mut stmt = conn.prepare_cached("INSERT INTO property (county, state, FIPS, feature, fixed_value, mult_value, mult_type, adder, min_dist, max_dist, value, units, ord_year, last_updated, section, source, comments) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").unwrap();
    for result in rdr.records() {
        let record = result.unwrap();
        println!("{:?}", record);
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
    //let df = polars::io::csv::read::CsvReadOptions::default().with_has_header(true).try_into_reader_with_file_path(Some("sample.csv".into())).unwrap().finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        init_db("test").unwrap();
    }
}
