//! Parse and handle the scrapped ordinance information

mod qualitative;
mod quantitative;

use tracing::trace;

use crate::error::Result;

#[derive(Debug)]
pub(super) struct Ordinance {
    quantitative: quantitative::Quantitative,
    qualitative: qualitative::Qualitative,
}

impl Ordinance {
    pub(super) fn init_db(conn: &duckdb::Transaction) -> Result<()> {
        trace!("Initializing database for Ordinance");

        quantitative::Quantitative::init_db(conn)?;
        qualitative::Qualitative::init_db(conn)?;

        trace!("Database ready for Ordinance");
        Ok(())
    }

    /// Open the quantitative ordinance from scrapped output
    pub(super) async fn open<P: AsRef<std::path::Path>>(root: P) -> Result<Ordinance> {
        trace!("Opening quantitative ordinance of {:?}", root.as_ref());

        let ordinance = Ordinance {
            quantitative: quantitative::Quantitative::open(root.as_ref()).await?,
            qualitative: qualitative::Qualitative::open(root.as_ref()).await?,
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
