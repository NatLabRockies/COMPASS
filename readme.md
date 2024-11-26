# Versioned data for ordinance records

## Command line

### Windows (x86_64)

``` shell
Invoke-WebRequest -Uri 'https://github.com/NREL/NREL-ordinance-DB/releases/latest/download/nrel-ordinance-cli-x86_64-pc-windows-msvc.exe' -OutFile ordinance
```

### MacOS (Apple Silicon)

``` shell
curl -o ordinance -L https://github.com/NREL/NREL-ordinance-DB/releases/latest/download/nrel-ordinance-cli-aarch64-apple-darwin
chmod +x ordinance
```

### MacOS (Intel)

``` shell
curl -o ordinance -L https://github.com/NREL/NREL-ordinance-DB/releases/latest/download/nrel-ordinance-cli-x86_64-apple-darwin
chmod +x ordinance
```

### Linux (x86_64)

``` shell
curl -o ordinance -L https://github.com/NREL/NREL-ordinance-DB/releases/latest/download/nrel-ordinance-cli-x86_64-unknown-linux-gnu
chmod +x ordinance
```

