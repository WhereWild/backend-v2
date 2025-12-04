# WhereWild

UPDATE THIS AS WE MAKE CHANGES. DOCUMENT WORKFLOWS, SETUP, ARCHITECTURE, ETC

# FastAPI

FastAPI allows the frontend and backend to communicate. To get the API service live, simply make sure your Docker image is up to date and run `docker compose up gdal`. Then make sure you get a response at `http://127.0.0.1:8000/health`.

FOR FRONTEND CONNECTION: Prior to starting the server, ensure you have installed the species.zip file and have unzipped it into a folder called processed (such that the filepath looks like processed/species/species_catalog.json for the catalog). Then, run the command $env:SPECIES_DIR = "processed/species" and then start the server as normal. From there, you should be able to start the frontend and naviagate to species pages that will be populated by data from the backend.