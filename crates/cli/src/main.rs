use std::path::PathBuf;

use clap::{arg, command, value_parser, Arg, ArgAction, Command};
use tracing::trace;

fn main() {
    let matches = command!() // requires `cargo` feature
        .arg(arg!(--db <DATABASE>).required(true))
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
                .arg(Arg::new("path").value_parser(value_parser!(PathBuf))),
        )
        .subcommand(
            Command::new("export")
                .about("Export the database")
                .arg(arg!(--format <FORMAT>).default_value("csv")),
        )
        .subcommand(Command::new("log").about("Show the history of the database"))
        .get_matches();

    let verbose = matches.get_count("verbose");
    if verbose > 0 {
        trace!("verbose level: {:?}", verbose);
        println!("verbose level: {:?}", verbose);
    }

    //       Command::new("log")
    //          .about("Show the history of the database")
    let db = matches.get_one::<String>("db").expect("required");

    match matches.subcommand_name() {
        Some("init") => {
            trace!("Creating database at {:?}", &db);
            ordinance::init_db(db).unwrap();
        }
        Some("export") => {
            trace!("Showing export for database at {:?}", &db);
            if verbose > 0 {
                println!("Showing export for database at {:?}", &db);
            }

            ordinance::export_db(&db);
        }
        Some("log") => {
            trace!("Showing log for database at {:?}", &db);
        }
        _ => {
            println!("No subcommand was used");
        }
    }
}
