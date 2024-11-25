use clap::{arg, command, value_parser, Arg, Command};
use std::path::PathBuf;

fn main() {
    let matches = command!() // requires `cargo` feature
        .arg(arg!(--db <DATABASE>).required(true))
        .subcommand(Command::new("init").about("Initialize a new database"))
        .subcommand(
            Command::new("load")
                .about("Load ordinance raw data")
                .arg(Arg::new("path").value_parser(value_parser!(PathBuf))),
        )
        .subcommand(Command::new("log").about("Show the history of the database"))
        .get_matches();

    //       Command::new("log")
    //          .about("Show the history of the database")
    let db = matches.get_one::<String>("db").expect("required");
    println!("two: {:?}", db);

    if let Some(matches) = matches.subcommand_matches("init") {
        println!("Creating database at {:?}", "mane");
        ordinance::init_db(db).unwrap();
    } else if let Some(matches) = matches.subcommand_matches("log") {
        println!("Showing log for database at {:?}", "mane");
    }
}
