//! Parse and handle the scrapped qualitative ordinance information

use tracing::trace;

use crate::error::Result;

#[derive(Debug)]
pub(super) struct Qualitative(Vec<QualitativeRecord>);

#[allow(dead_code, non_snake_case)]
#[derive(Debug, serde::Deserialize)]
pub(super) struct QualitativeRecord {
    county: String,
    state: String,
    subdivison: Option<String>,
    jurisdiction_type: Option<String>,
    FIPS: u64,
    feature: String,
    summary: String,
    ord_year: Option<u32>,
    section: Option<String>,
    source: Option<String>,
}

impl Qualitative {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        trace!("Initializing database for Qualitative Ordinance");

        conn.execute_batch(
            r"
            CREATE SEQUENCE IF NOT EXISTS qualitative_sequence START 1;
            CREATE TABLE IF NOT EXISTS qualitative (
              id INTEGER PRIMARY KEY DEFAULT
                NEXTVAL('qualitative_sequence'),
              bookkeeper_lnk INTEGER REFERENCES bookkeeper(id) NOT NULL,
              county TEXT,
              state TEXT,
              subdivison TEXT,
              jurisdiction_type TEXT,
              FIPS UBIGINT,
              feature TEXT,
              summary TEXT,
              ord_year INTEGER,
              section TEXT,
              source TEXT
            );",
        )?;

        trace!("Database ready for Qualitative Ordinance");
        Ok(())
    }

    /// Open the qualitative ordinance from scrapped output
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Qualitative> {
        trace!("Opening qualitative ordinance of {:?}", root.as_ref());

        let path = root.as_ref().join("qualitative_ordinances.csv");
        if !path.exists() {
            trace!("Missing qualitative ordinance file: {:?}", path);
            return Err(crate::error::Error::Undefined(
                "Missing qualitative ordinance file".to_string(),
            ));
        }

        trace!("Identified qualitative ordinance at {:?}", path);

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

        trace!("Qualitative reader {:?}", rdr);

        let mut output = Vec::new();
        for result in rdr.deserialize() {
            let record: QualitativeRecord = match result {
                Ok(record) => record,
                Err(_) => {
                    trace!("Error {:?}", result);
                    continue;
                }
            };
            output.push(record);
        }
        trace!("Qualitative ordinance records {:?}", output);

        Ok(Qualitative(output))
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        trace!("Writing qualitative ordinance to database");

        let mut stmt = conn
            .prepare(
                r"INSERT INTO qualitative
            (bookkeeper_lnk, county, state, subdivison,
            jurisdiction_type, FIPS, feature, summary, ord_year,
            section, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )
            ",
            )
            .expect("Failed to prepare qualitative ordinance statement");

        for record in &self.0 {
            trace!("Writing qualitative ordinance record {:?}", &record);
            stmt.execute(duckdb::params![
                commit_id,
                record.county,
                record.state,
                record.subdivison,
                record.jurisdiction_type,
                record.FIPS,
                record.feature,
                record.summary,
                record.ord_year,
                record.section,
                record.source,
            ])?;
        }

        trace!("Qualitative Ordinance written to database");
        Ok(())
    }
}

#[cfg(test)]
/// Samples of qualitative ordinance to support testing
pub(crate) mod sample {
    use crate::error::Result;
    use std::io::Write;

    pub(crate) fn basic() -> String {
        let mut output = String::new();
        output.push_str("county,state,subdivison,jurisdiction_type,FIPS,feature,summary,ord_year,section,source\n");
        output.push_str(
            "county-1,state-1,,jurisdiction_type-1,11111,feature-1,summary-1,2001,section-1,source-1\n",
        );
        output.push_str(
            "county-2,state-2,,jurisdiction_type-2,22222,feature-2,summary-2,2002,section-2,source-2\n",
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
        let _file = sample::as_file(tmp.path().join("qualitative_ordinances.csv")).unwrap();

        let ord = Qualitative::open(&tmp).await.unwrap();
        dbg!(&ord);
        //assert_eq!(&ord.0[0].county, "county-1");
        //assert_eq!(&ord.0[0].feature, "feature-1");
    }
}
