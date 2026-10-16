# Raw collection storage

`seed_urls.csv` is the checked-in list of official pages the collector is allowed to request. It lists Coachella, Bonnaroo, Lollapalooza, EDC Las Vegas, and Outside Lands.

HTML saved during a run goes in `html/` and stays local. This directory is ignored by git except for this README, `.gitkeep`, and `seed_urls.csv`.

Collected documents intended for retrieval are written to `../processed/documents.csv` by `src/collect.py`. That CSV is local coursework data copied from official pages. Do not commit or redistribute it unless the source sites permit reuse.
