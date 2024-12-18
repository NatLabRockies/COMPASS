# Developer Guide

A guide for developers who want to contribute to the project.
I'll just collect some useful information here for now, so I don't forget it, and organize it later.

- Because we statically link the requirements, the compilation process can
  extend to 30-60 minutes. That is mostly due to duckdb and tokio. To
  optimize this process, we use GA cache, but too many images blow the quota
  make the process less useful. To avoid this, the Rust component is setup
  to only save the compiled cache for branch main, and all other branches
  initiate from that, and hopefully save some fair amount of time. Thus,
  a good practice here is to update Rust dependencies in a dedicated branch
  to avoid too much recompilation in working branches.
