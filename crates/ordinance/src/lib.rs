/// NREL's ordinance database
use duckdb::Connection;

pub fn init_db(path: &str) -> duckdb::Result<()> {
    let db = Connection::open(&path)?;
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
      geometry GEOMETRY NOT NULL,
      rank jurisdiction_rank NOT NULL,
      parent_id INTEGER REFERENCES jurisdiction(id),
      created_at TIMESTAMP NOT NULL DEFAULT NOW(),
      src TEXT,
      comments TEXT
      );

    COMMIT;",
    )?;

    println!("{}", db.is_autocommit());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        init_db("mane").unwrap();
    }
}
