astra
=====

The analysis framework for the Sloan Digital Sky Survey.

Astra is the analysis framework for the Sloan Digital Sky Survey (SDSS-V) Milky
Way Mapper. The purpose of Astra is to manage the analysis of reduced data
products from SDSS and to streamline data releases.

Installation
------------

I recommend you use `uv` to create an environment for Astra. Download the repository and install from source:
```
git clone git@github.com:sdss/astra.git
cd astra
uv venv
source .venv/bin/activate
uv pip install -e .
```

Setup
-----

You will need to initialize a database for Astra. A PostgreSQL database is recommended. The database connection details need to be stored in a `~/.astra/astra.yml` file in the following format:

```
database:
  schema: <schema_name>
  dbname: <database_name>
  user: <user_name>
  host: <host_name>
```

If you're using a specific schema, you'll need to create that schema first, then run:

```
astra init
``` 

Astra does not need to be run in the same computing environment where the data are stored. For this reason, it needs to _migrate_ details about available spectra into the Astra database, and link all the auxillary data (photometry, astrometry) for those sources. You can do this using the `astra migrate` tool.

```
astra migrate
```

Here's what it looks like:

![Migrations](./docs/astra-migrate-2024-11-14.gif)


