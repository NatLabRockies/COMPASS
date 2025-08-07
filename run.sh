printf "%b\n" "$JURISDICTIONS" > jurisdictions.csv
printf "%b\n" "$COMPASS_CONFIG" > config.json5
compass process -c config.json5 -np -vv
