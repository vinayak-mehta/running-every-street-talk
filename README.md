# Running every street in Paris with Python and PostGIS

Slides and code for my talk on running every street in Paris and automatically tracking my progress.

The approach - fetch the street network from OpenStreetMap, draw routes, run them, map match the GPX data to the street network, and visualize coverage.

See current progress at [dub.sh/everystreet](https://dub.sh/everystreet).

## Notebooks

- `streets.ipynb` - fetching and storing the Paris street network from OSM using osmnx and PostGIS
- `matching.ipynb` - map matching GPX runs to the street network and visualizing progress

## Setup

Requires PostgreSQL with the PostGIS extension enabled:

```sql
CREATE EXTENSION postgis;
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or install from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## Slides

- [streets.slides.html](streets.slides.html) - street network and PostGIS
- [matching.slides.html](matching.slides.html) - map matching and visualization
- [slides.pdf](slides.pdf) - full talk slides
