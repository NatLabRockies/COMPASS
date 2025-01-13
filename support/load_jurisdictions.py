
import duckdb
import geopandas

df = geopandas.read_file("ord_areas.gpkg")
db = duckdb.connect("ROSIE.db")

bookkeepping_id = db.sql("INSERT INTO bookkeeping (hash) values ('dummy_hash') RETURNING id").fetchone()[0]

for _, per_state in df.groupby(["state_fips"]):
    state = per_state[["state_name", "geometry"]].dissolve(by="state_name")
    state = state.merge(per_state[["state_name", "state_fips"]].drop_duplicates(), on="state_name").iloc[0]

    state_pk = db.execute(f"INSERT INTO jurisdiction (name, bookkeeping_lnk, fips, rank, geometry) values ('{state.state_name}', {bookkeepping_id}, {state.state_fips}, 'state', '{state.geometry}') RETURNING id").fetchone()[0]

    county = per_state[["state_name", "state_fips", "county_name", "county_fips", "geometry"]].dropna()
    for _, site in county.iterrows():
        db.execute(f"INSERT INTO jurisdiction (bookkeeping_lnk, parent_id, name, FIPS, rank, geometry) values ({bookkeepping_id}, {state_pk}, '{site.county_name.replace("'", "''''")}', {site.county_fips}, 'county', '{site.geometry}')")

    subd = per_state[["state_name", "state_fips", "subd_name", "subd_fips", "geometry"]].dropna()
    for _, site in subd.iterrows():
        if site.subd_name.endswith('town'):
            rank = "town"
        elif site.subd_name.endswith('city'):
            rank = "city"
        elif site.subd_name.endswith('CCD'):
            rank = "CCD"
        elif site.subd_name.endswith('Reservation'):
            rank = "reservation"
        else:
            rank = "other"
            print(site.subd_name)
        db.execute(f"INSERT INTO jurisdiction (bookkeeping_lnk, parent_id, name, FIPS, rank, geometry) values ({bookkeepping_id}, {state_pk}, '{site.subd_name.replace("'", "''''")}', {site.subd_fips}, '{rank}', '{site.geometry}')")

