from astra import config, log

def get_profile():
    try:
        profile = config["migrations"]["sdss5db"]["profile"]
    except:
        profile = "pipelines"
        #log.warn(f"No `profile` found in Astra's config `migrations.sdss5db.profile`. Using '{profile}' profile.")
    return profile


def get_approximate_rows(model):
    return int(
        model._meta.database.execute_sql(
            f"SELECT reltuples FROM pg_class WHERE relname = '{model.__name__.lower()}';"
        )
        .fetchone()[0]
    )
    