use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, Command, arg, command, value_parser};
use duckdb::Connection;
use tracing::{self, error, info, trace};

fn main() -> Result<()> {
    let matches = command!() // requires `cargo` feature
        .arg(
            arg!(--db <DATABASE>)
                .required(true)
                .help("Path to the database file. Ex.: ./ordinance.db"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .action(ArgAction::Count)
                .help("Set the verbosity level, ex.: -vvv"),
        )
        .subcommand(Command::new("init").about("Initialize a new empty database"))
        .subcommand(
            Command::new("load")
                .about("Load ordinance raw data")
                .arg(
                    Arg::new("username")
                        .short('u')
                        .required(true)
                        .help("Username to use"),
                )
                .arg(
                    Arg::new("path")
                        .value_parser(value_parser!(PathBuf))
                        .required(true)
                        .help("Path to directory with scraper output"),
                ),
        )
        .subcommand(
            Command::new("export")
                .about("Export the database")
                .arg(
                    Arg::new("OUTPUT")
                        .required(true)
                        .short('o')
                        .long("output")
                        .value_parser(value_parser!(PathBuf))
                        .help("Path to the output directory, ex.: './ordinance_export'"),
                )
                .arg(
                    Arg::new("TECHNOLOGY")
                        .short('t')
                        .long("technology")
                        .required(true)
                        .value_parser(["wind", "solar"])
                        .help("Technology to export, ex.: 'wind'"),
                )
                .arg(
                    Arg::new("FORMAT")
                        .short('f')
                        .long("format")
                        .help("Format to export, ex.: 'csv' or 'json'")
                        .value_parser(["csv", "revx", "json", "gpkg", "gpq"])
                        .default_value("revx")
                        .default_missing_value("revx"),
                ),
        )
        .subcommand(Command::new("log").about("Show the history of the database"))
        .get_matches();

    let verbose = matches.get_count("verbose");
    let tracing_level = match verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .init();
    info!("Verbosity level: {:?}", verbose);

    //       Command::new("log")
    //          .about("Show the history of the database")
    let db = matches.get_one::<String>("db").expect("required");

    match matches.subcommand_name() {
        Some("init") => {
            trace!("Creating database at {:?}", &db);
            infra_compass_db::init_db(db)
                .with_context(|| format!("Failed to initialize database as {db}"))?;
        }
        Some("export") => {
            trace!("Exporting database {:?}", &db);

            let technology = matches
                .subcommand_matches("export")
                .unwrap()
                .get_one::<String>("TECHNOLOGY")
                .unwrap();
            trace!("Filtering technology: {:?}", &technology);

            let format = matches
                .subcommand_matches("export")
                .unwrap()
                .get_one::<String>("FORMAT")
                .unwrap();
            trace!("Output format: {:?}", &format);

            let output = matches
                .subcommand_matches("export")
                .unwrap()
                .get_one::<PathBuf>("OUTPUT")
                .unwrap();
            trace!("Output to: {:?}", &output);
            if output.exists() {
                error!(
                    "Output {:?} already exists. Please remove it before exporting.",
                    output
                );
                anyhow::bail!("Output already exists");
            }

            let mut wrt = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(output)
                    .expect("Failed to open output file"),
            );
            trace!("Output file created: {:?}", &wrt);

            infra_compass_db::export(&mut wrt, db, format, technology)?;
        }
        Some("load") => {
            trace!("Subcommand load");
            trace!("Using database: {:?}", &db);
            let username = matches
                .subcommand_matches("load")
                .unwrap()
                .get_one::<String>("username")
                .unwrap();
            trace!("Username: {:?}", &username);
            let path = matches
                .subcommand_matches("load")
                .unwrap()
                .get_one::<PathBuf>("path")
                .unwrap();
            trace!("Loading data from: {:?}", &path,);

            // In the future, replace this Connection with a custom one
            // that already creates a session with the username, and hance
            // handle ahead permissions/authorization.
            let conn: Connection = Connection::open(db).expect("Failed to open database");
            infra_compass_db::load_ordinance(conn, username, path).with_context(|| {
                format!("Failed to load ordinance data from {}", path.display(),)
            })?;
        }

        Some("log") => {
            trace!("Showing log for database at {:?}", &db);
        }
        _ => {
            println!("No subcommand was used");
        }
    }

    Ok(())
}
