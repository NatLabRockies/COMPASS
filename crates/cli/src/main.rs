use std::path::PathBuf;

use clap::{Arg, ArgAction, Command, arg, command, value_parser};
use duckdb::Connection;
use tracing::{self, trace};
use tracing_subscriber;

fn main() {
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
                        .help("Path to directory with scrapper output"),
                ),
        )
        .subcommand(
            Command::new("export")
                .about("Export the database")
                .arg(arg!(--format <FORMAT>).default_value("csv")),
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
    tracing::info!("Verbosity level: {:?}", verbose);

    //       Command::new("log")
    //          .about("Show the history of the database")
    let db = matches.get_one::<String>("db").expect("required");

    match matches.subcommand_name() {
        Some("init") => {
            trace!("Creating database at {:?}", &db);
            infra_compass_db::init_db(db).unwrap();
        }
        Some("export") => {
            trace!("Showing export for database at {:?}", &db);
            if verbose > 0 {
                println!("Showing export for database at {:?}", &db);
            }

            infra_compass_db::export_db(&db);
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
            let conn: Connection = Connection::open(&db).expect("Failed to open database");
            let _ = infra_compass_db::load_ordinance(conn, username, path);
        }
        Some("log") => {
            trace!("Showing log for database at {:?}", &db);
        }
        _ => {
            println!("No subcommand was used");
        }
    }
}
