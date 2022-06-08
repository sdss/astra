# Data Products

If a task does some work on a data product (e.g., an observed spectrum of a star) then it needs to know where the observed data are. Similarly, if your task produces some data products, then other people (or tasks) need to know where those data products are.

## Input data products

In theory you could define a parameter for your task like `observed_data`. 
This is a useful thing to do when testing your code so that you don't have to do any extra work.
But if you want your code to run on large amounts of data, it's a better idea to record the input data paths in a different way so that you can easily cross-match results from different tasks on the same input data, or link data paths to the same source object (e.g., different observations of the same star).

We record these input data products as {obj}`astra.database.astradb.DataProduct` records. Each `DataProduct` object has a unique primary key, and has enough information to reconstruct the data path. The word *reconstruct* is important here, as you will see below, but first let's start with a really simple example:

```python
from astra.database.astradb import DataProduct

# Get or create a new data product.
observed_data, created = DataProduct.get_or_create(
    filetype="full",
    kwargs=dict(
        full="my_observed_data.csv"
    )
)

print(observed_data)
# <DataProduct: id=1>
print(observed_data.path)
# my_observed_data.csv
```

Every task instance has an optional keyword argument called `input_data_products` where you can provide a list of data products. 
When you use this method to supply the input data products to your task, a reference is created in the Astra database that links the data products and tasks together, allowing you to easily query which tasks have executed on what data products, or vice versa. 
Specifically, a record is created in the {obj}`astra.database.astradb.TaskInputDataProducts` table, which allows for a many-to-many relationship between tasks and data products.
Usually a task might only have one input data product, and that's OK. In that case you should still use the `input_data_products` (note the plural) keyword. Once you've tested your code, this is the right way to supply data products to tasks.

Now let's create a task that uses this data product:

```python
from astropy.table import Table
from astra.base import TaskInstance

class EstimateSNR(TaskInstance):

    def execute(self):
        snr = []
        for data_product in self.input_data_products:
            data = Table.read(data_product.path)
            snr.append(np.mean(data["flux"]/data["flux_error"]))
        
        return snr

task = EstimateSNR(input_data_products=[observed_data])
task.execute()
```


## Output data products

If your task creates data products that will be used by end-users or by other tasks, then it's good practice for your task to create a {obj}`astra.database.astradb.DataProduct` for those outputs. 
The way this is achieved is by creating an entry in the {obj}`astra.database.astradb.TaskOutputDataProducts` table, which acts as a many-to-many join between tasks and data products.

Linking the input and output data products in this way lets us efficiently query to see how intermediate data products are used between tasks. In SDSS, this could be intermediate data products in a complex pipeline like ASPCAP. Or it could be a trained Cannon model -- the output data product from a `TrainTheCannon` task -- where the input data products to the `TrainTheCannon` model were thousands of ApStar spectra. Linking the data products like this lets us easily ask questions like: was this random SDSS star part of the training set used for that model?

```{todo}

TO DO: add code example for linking output data products
```


## Data products and access paths

The {obj}`astra.database.astradb.DataProduct` references a database record for a data product, which we earlier said we could use to *reconstruct* the path. If you are not using SDSS data, then this part is not relevant to you.

Every SDSS product released must have a corresponding 'data model' that defines the path to the file and describes its form and content. 
If you want to reference a data product then instead of using the path, you can store the variables that describe the path. For example, in SDSS-V if you want to retrieve an 'ApStar' spectrum then you need to know the following keywords:
- `healpix`: the identifier of the healpix bin that the star position (right ascension and declination) falls in
- `apred`: the version of the APOGEE reduction pipeline used
- `telescope`: the name of the telescope that took the observations
- `obj`: the 2MASS-like object identifier (e.g., 2M000000+000000)
- `apstar`: a keyword argument used for differentiating test data from real data. This is usually 'star'.

You can confirm this for yourself with the `sdss_access` code:

```python
from sdss_access import SDSSPath

release = "sdss5"
filetype = "apStar"

print(SDSSPath(release=release).lookup_keys(filetype))
# ['healpix', 'apred', 'telescope', 'obj', 'apstar']

print(SDSSPath(release=release).templates[filetype])
# '$APOGEE_REDUX/{apred}/{apstar}/{telescope}/@healpixgrp|/{healpix}/apStar-{apred}-{telescope}-{obj}.fits'
```

With these five (or four) keywords, you can identify the path of the file. When a {obj}`astra.database.astradb.DataProduct` is created we require the following keywords:
- `release`: defaults to `None`
- `filetype`: the file type, same usage as per `sdss_access`
- `kwargs`: the keyword arguments that define the file path

In the earlier example we only supplied `filetype="full"` and `kwargs` was a dictionary: `kwargs=dict(full="...")`. That's because the path of every {obj}`astra.database.astradb.DataProduct` is resolved by using {obj}`sdss_access.SDSSPath`, and the 'full' filetype is a special kind that does not depend on the `release`, it is a way of directly supplying full file paths.

If you can, you should create data products with the appropriate file type and keywords. That means you should do something like this:

``````{tab} Good (do this)
```python
from sdss_access import SDSSPath
from astra.database.astradb import DataProduct

release, filetype = ("sdss", "apStar")
kwargs = dict(
    healpix=5000,
    apred="daily",
    telescope="apo25m",
    obj="2M112354-301245",
    apstar="star"
)

# This is good practice: it means the data product is stored in the
# database as an 'apStar' file, and the keyword arguments make it
# easier for us to query data products by object later on.
data_product, _ = DataProduct.get_or_create(
    release=release,
    filetype=filetype,
    kwargs=kwargs
)
```
``````
``````{tab} Bad (avoid this)
```python
from sdss_access import SDSSPath
from astra.database.astradb import DataProduct

release, filetype = ("sdss", "apStar")
kwargs = dict(
    healpix=5000,
    apred="daily",
    telescope="apo25m",
    obj="2M112354-301245",
    apstar="star"
)

# BAD STARTS HERE
path = SDSSPath(release=release).full(filetype, **kwargs)

# This is BAD because we could have created the data product with the
# 'apStar' file type, which would have been easier to later query 
# data products by object name, healpix, etc.

data_product, _ = DataProduct.get_or_create(
    filetype="full",
    kwargs=dict(
        full=path
    )
)
```
``````

While you're testing things, it's fine to use the `full` filetype for data products your code has created.
It's also acceptable to use the `full` filetype for intermediate data products that your code has created: for things that users won't ever use.
If you're creating data products for end-users, you should write a data model for it and merge it into the [SDSS `tree` product](https://github.com/sdss/tree).