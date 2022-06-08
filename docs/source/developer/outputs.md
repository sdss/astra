# Outputs

Many tasks will produce results (e.g., measurements of things) that you want to store in a database, not just store in [output data products](data-products).

The typical use-case for Astra is that many different methods might be used to make measurements or estimate parameters, and you want to compare those measurements easily. 
You could just write everything to data products and open them later on to compare, but that can be very inefficient.
It's a better idea to store some summary results in a database.


Each task will have different things that it reports. 
For this reason, if you want a task to write specific outputs then you should create a database table with all the outputs
that you expect.
We'll show an example of how to do that on this page.
With many different tasks, that means you might have 10 (or more) database tables with outputs from different tasks. 
We want to be able to uniquely identify each output, without necessarily knowing which database table it is stored in.

For this reason, Astra uses a few database tables to link things together. Let's imagine your analysis method is called "Rocket", and the outputs of your task are going to be stored in the Astra database as `astra.database.astradb.RocketOutput`. 
The executed task instance details are stored in {obj}`astra.database.astradb.Task`.
To create an output, we first create a {obj}`astra.database.astradb.Output` record, which gives us a unique identifier for the output. 
Tasks and outputs are linked by a many-to-many relationship through the {obj}`astra.database.astradb.TaskOutput` table. 
When we create our output in `astra.database.astradb.RocketOutput`, that table will have a foreign reference to the {obj}`astra.database.astradb.Output` record.

This design allows us to:
- write bespoke database tables for task outputs without having to be strict about which column names to use,
- have a unique reference to every output, regardless of how many output tables there are,
- access the outputs of a task without knowing *a priori* what the database table is,
- link many outputs to one task, and 
- link one output to many tasks (if desired).


## Adding a database table for task outputs

If you have a task that is going to produce useful outputs for end-users, which should be kept in the database,
then you should write your own database table. In the [Astra GitHub repository](https://github.com/sdss/astra)
all Astra database models are defined in `python/astra/database/astradb.py`.
That's the file you will need to edit to add your database table. Your database table should be a sub-class of {obj}`astra.database.astradb.AstraOutputBaseModel`, which is defined in the same file.

Here's an example data model for the outputs of the primary task in "Rocket":

```python
# python/astra/database/astradb.py
# Note: The field classes FloatField, IntegerField, etc are already imported

class RocketOutput(AstraOutputBaseModel):

    """ Summary outputs from Rocket. """

    snr = FloatField()
    teff = FloatField()
    logg = FloatField()
    fe_h = FloatField()
    u_teff = FloatField()
    u_logg = FloatField()
    u_fe_h = FloatField()
    bitmask_flag = IntegerField(default=0)
```

Astra uses `u_` prefix to indicate uncertainty. 
If you added this code to `python/astra/database/astradb.py` and re-installed Astra,
your `RocketOutput` database table would be created next time you run:
```bash
astra initdb
```

Because `RocketOutput` is a sub-class of `AstraOutputBaseModel`, your database table will include a few more column names that are common to all `*Output` tables:
- `output_id`: a foreign key reference to {obj}`astra.database.astradb.Output.id`
- `task_id`: a foreign key reference to {obj}`astra.database.astradb.Task.id`
- `source_id`: a foreign key reference to {obj}`astra.database.astradb.Source.id`
- `meta`: an optional JSON-like field for storing any metadata. 

The `*_id` foreign key references are accessible through `RocketOutput.output`, `RocketOutput.task`, and `RocketOutput.source`. In principle the only reference we need is through `RocketOutput.output`: the `task` and `source` references are only for convenience.

JSON fields like `meta` are relatively expensive, so if you know what metadata fields you want to store ahead of time, then you should put these in as explicit columns instead of storing them in `meta`. For example, `snr` is a metadata-like field that we added to `RocketOutput` because we know we will estimate the signal-to-noise ratio (SNR) each time a task instance is executed. `meta` is useful for situations where the column names you need to store are different each time, or will only be used some of the time. 

```{note}
Check out the [software guide](software-index) before making code changes.
```

## A task that writes an output


```{todo} TODO
Need some nice functions for creating task outputs that will keep them indempodent. For now you will need to create the `Output` object yourself. As a minimal example, check out {obj}`astra.contrib.apogeenet.base.StellarParameters`.
```

## Keep tasks indempodent


Now that you know how to write a task that can [take in parameters](parameters), read and write [data products](data-products), and create summary outputs, it's time to learn about how tasks can be [bundled together](bundles) for efficiency.
