//! Parse and handle the scrapped quantitative ordinance information

use tracing::trace;

use crate::error::Result;

#[derive(Debug)]
pub(super) struct Quantitative(Vec<QuantitativeRecord>);

#[allow(dead_code, non_snake_case)]
#[derive(Debug, serde::Deserialize)]
pub(super) struct QuantitativeRecord {
    county: String,
    state: String,
    subdivison: Option<String>,
    jurisdiction_type: Option<String>,
    FIPS: u64,
    feature: String,
    value: f64,
    units: Option<String>,
    offset: Option<f64>,
    min_dist: Option<f64>,
    max_dist: Option<f64>,
    summary: Option<String>,
    ord_year: Option<u32>,
    section: Option<String>,
    source: Option<String>,
}

impl Quantitative {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        trace!("Initializing database for Quantitative");

        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS quantitative_sequence START 1;
            CREATE TABLE IF NOT EXISTS quantitative (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('quantitative_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              county TEXT,
              state TEXT,
              subdivison TEXT,
              jurisdiction_type TEXT,
              FIPS UBIGINT,
              feature TEXT,
              value REAL,
              units TEXT,
              adder REAL,
              min_dist REAL,
              max_dist REAL,
              summary TEXT,
              ord_year INTEGER,
              section TEXT,
              source TEXT
            );",
        )?;

        trace!("Database ready for Quantitative");
        Ok(())
    }

    /// Open the quantitative ordinance from scrapped output
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Quantitative> {
        trace!("Opening quantitative ordinance of {:?}", root.as_ref());

        let path = root.as_ref().join("quantitative_ordinances.csv");
        if !path.exists() {
            trace!("Missing quantitative ordinance file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing quantitative ordinance file".to_string(),
            ));
        }

        trace!("Identified quantitative ordinance at {:?}", path);

        /*
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(path.into()).unwrap()
            .finish()
            .unwrap();
        */

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b',')
            .from_path(&path)
            .unwrap();

        trace!("Quantitative reader {:?}", rdr);

        let mut output = Vec::new();
        for result in rdr.deserialize() {
            let record: QuantitativeRecord = match result {
                Ok(record) => record,
                Err(_) => {
                    trace!("Error {:?}", result);
                    continue;
                }
            };
            output.push(record);
        }
        trace!("Quantitative ordinance records {:?}", output);

        Ok(Quantitative(output))
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        trace!("Writing ordinance to database");

        let mut stmt = conn
            .prepare(
                r"INSERT INTO quantitative
            (bookkeeper_lnk, county, state, subdivison,
            jurisdiction_type, FIPS, feature, value, units, adder,
            min_dist, max_dist, summary, ord_year, section, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
            )
            .expect("Failed to prepare ordinance statement");

        for record in &self.0 {
            trace!("Writing ordinance record {:?}", &record);
            stmt.execute(duckdb::params![
                commit_id,
                record.county,
                record.state,
                record.subdivison,
                record.jurisdiction_type,
                record.FIPS,
                record.feature,
                record.value,
                record.units,
                record.offset,
                record.min_dist,
                record.max_dist,
                record.summary,
                record.ord_year,
                record.section,
                record.source,
            ])?;
        }

        trace!("Quantitative written to database");
        Ok(())
    }
}

#[cfg(test)]
/// Samples of quantitative ordinance to support testing
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    pub(crate) fn basic() -> String {
        let mut output = String::new();
        output.push_str("county,state,subdivison,jurisdiction_type,FIPS,feature,value,units,offset,min_dist,max_dist,summary,ord_year,section,source\n");
        output.push_str(
            "county-1,state-1,,jurisdiction_type-1,11111,feature-1,,,,,,,2001,,source-1\n",
        );
        output.push_str(
            "county-2,state-2,,jurisdiction_type-2,22222,feature-2,,,,,,,2002,,source-2\n",
        );
        output
    }

    pub(crate) fn as_file<P: AsRef<std::path::Path>>(path: P) -> Result<std::fs::File> {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "{}", basic())?;
        file.flush()?;
        Ok(file)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn dev() {
        let tmp = tempfile::tempdir().unwrap();
        let _file = sample::as_file(tmp.path().join("quantitative_ordinances.csv")).unwrap();

        let ord = Quantitative::open(&tmp).await.unwrap();
        dbg!(&ord);
        //assert_eq!(&ord.0[0].county, "county-1");
        //assert_eq!(&ord.0[0].feature, "feature-1");
    }
}
