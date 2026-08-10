# COMPASS Jurisdiction Compilation

This document briefly outlines the process used to compile the known CONUS jurisdictions data for COMPASS.

## Steps

1. Run the `compile_jurisdiction_gpkg.ipynb` notebook. This pulls data from the CENSUS for states, counties, subdivisions, and places. The data is cleaned and combined into one collection of jurisdictions.

2. Run the `find_website_using_wiki.py` script. This pulls website info for as many jurisdictions as possible from the wiki. At this point, the websites have not been validated, so the data is not ready to use yet.

3. Run the `update_jur_websites.py` script. This attempts to find any remaining websites using a search engine and verifies the sites found in the previous step.

After running these steps, the website information should be ready for use.
