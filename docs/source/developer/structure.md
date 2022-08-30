# Structure

This page describes how you should structure any code that you want to add to Astra.

## Contributed components

The contributed components in Astra all live in the {obj}`astra.contrib` namespace.
Let's imagine you wanted to add a component called 'Rocket'.

This component will have some tasks that write results to the database.
The outputs in the database will need their own database table, `RocketOutput`, so we will need to add a data model for this in `python/astra/database/astradb.py`.
Let's imagine that Rocket has some *new* method for continuum normalisation, called "Takeoff", which we will add to Astra in a way so that other codes can also use it. To do that, we will add the normalisation code in `python/astra/tools/continuum/takeoff.py`
 (to a new table in the database),
and imagine this Rocket code has a "new" method for continuum normalisation (called "Takeoff"), which we will add to Astra in a way so that other codes could also use it.

From the [Astra GitHub repository](https://github.com/sdss/astra), the relevant folder and file structure might look something like this:

```
python/
    astra/
        contrib/
            rocket/
                __init__.py
                base.py
                model.py
                utils.py

        database/
            astradb.py

        tools/
            continuum/
                takeoff.py
```

Within the `rocket/` directory, `base.py` is where we will define task instances. We will define anything related to the model in `model.py`, and put utilities in `utils.py`. This is just a guide: the most important things are probably to keep this folder structure, and to put the task instances in `base.py`.
