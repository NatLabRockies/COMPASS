//! Parse and handle the scrapped ordinance information

mod qualitative;
mod quantitative;

use tracing::{debug, trace};

use crate::error::Result;

#[derive(Debug)]
pub(super) struct Ordinance {
    quantitative: quantitative::Quantitative,
    qualitative: qualitative::Qualitative,
}

impl Ordinance {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        debug!("Initializing database for Ordinance");

        quantitative::Quantitative::init_db(conn)?;
        qualitative::Qualitative::init_db(conn)?;

        trace!("Creating ordinance view combining quantiative and qualitative data");
        // Adding bookkeeper_lnk to allow linking with technology for now,
        // but this will change in the future.
        conn.execute_batch(
            r"
            CREATE VIEW IF NOT EXISTS ordinance AS
              SELECT bookkeeper_lnk, FIPS, feature, NULL as feature_subtype,
                value AS 'quantitative', NULL AS 'qualitative'
              FROM quantitative
              UNION
                SELECT bookkeeper_lnk, FIPS, feature, NULL as feature_subtype,
                  NULL AS 'quantitative', summary AS 'qualitative'
                FROM qualitative;",
        )?;

        trace!("Database ready for Ordinance");
        Ok(())
    }

    /// Open the quantitative ordinance from scrapped output
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Ordinance> {
        debug!("Opening ordinance from {:?}", root.as_ref());

        let (quantitative, qualitative) = tokio::try_join!(
            quantitative::Quantitative::open(root.as_ref()),
            qualitative::Qualitative::open(root.as_ref())
        )?;
        let ordinance = Ordinance {
            quantitative,
            qualitative,
        };

        trace!("Opened ordinance: {:?}", ordinance);

        Ok(ordinance)
    }

    pub(super) fn write(&self, conn: &duckdb::Transaction, commit_id: usize) -> Result<()> {
        trace!("Writing ordinance to database");

        self.quantitative.write(conn, commit_id)?;
        self.qualitative.write(conn, commit_id)?;

        trace!("Ordinance written to database");
        Ok(())
    }
}

#[cfg(test)]
/// Samples of quantitative ordinance to support testing
pub(crate) mod sample {
    use super::*;

    pub(crate) fn as_file<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
        let _quantitative =
            quantitative::sample::as_file(path.as_ref().join("quantitative_ordinances.csv"))
                .unwrap();
        let _qualitative =
            qualitative::sample::as_file(path.as_ref().join("qualitative_ordinances.csv")).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn dev() {
        let tmp = tempfile::tempdir().unwrap();
        sample::as_file(tmp.path()).unwrap();
        let _ordinance = Ordinance::open(&tmp).await.unwrap();
    }
}
