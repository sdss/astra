# Development Guide

This page covers the fundamentals of `astra` for a scientist or developer who wants to add an analysis component in `astra`.

## Task Instance

The base component of `astra` is a {obj}`astra.base.TaskInstance`. A task instance will perform some analysis on data products, write results to a database, and/or create output data products. Every {obj}`astra.base.TaskInstance` (or subclass thereof) has some parameter(s) that you define in the class definition. The actual work to do by a task is specified by the `execute()` method:

```python
from astra.base import TaskInstance, Parameter

class GreedyPrimes(TaskInstance):

    min_value = Parameter(default=2)
    max_value = Parameter()

    def execute(self):
        primes = []
        for n in range(self.min_value, self.max_value):
            for i in range(2, n):
                if n%i == 0: break
            else:
                primes.append(n)
        return primes
```

The `GreedyPrimes` task has only two parameters: `min_value` and `max_value`. The `min_value` parameter has a default value, so we only require a `max_value` to create an instance of `GreedyPrimes`. All parameters must be explicitly given as keyword arguments. If you don't supply a required parameter, a `TypeError` will be raised.

``````{tab} Python
```python

ti = GreedyPrimes(max_value=30)
primes = ti.execute()
print(primes)
```
``````
``````{tab} Output
```bash
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```
``````


## Why bother?

It looks like we just wrapped a function in a class for no reason. When we ran `GreedyPrimes.execute()` there were a few things that happened in the background. Specifically:

- A record of the task (`GreedyPrimes`) was recorded in the `astra` database.
- The task parameters were also recorded:
    + `min_value`: 2 
    + `max_value`: 30
- The version of `astra` was recorded with the task, as well as some other metadata for reproducibility and provenance.
- The time taken to perform different parts of `GreedyPrimes` is recorded in the database, for tracking performance.

If your analysis task writes to the database, or creates data products, then these outputs are linked in the database to the task itself. That means for every analysis output, you know exactly what the parameters were that went into that analysis, and other performance metadata.

This is done quietly in the background for every task instance. And any task instance recorded in the database can be re-constructed later on. For example, if you ran some expensive analysis task as part of a large pipeline and then later wanted to interrogate the internals carefully, you can create that task instance from the database:

```python
from astra.base import TaskInstance
from astra.database.astradb import Task

task = Task.get_by_id(666) # my suspicious task

ti = TaskInstance.from_task(task)

# execute the task interactively with some additional debug keywords
ti.execute(debug=True, verbose=True)
```


## Data products

A data product might be an observation of a single star (e.g., a SDSS apVisit file), or it could be a pickle file produced by someone else. 

When a task performs some analysis on a data product, the details of the data product are recorded in the database and linked to that task. Similarly, when a task creates an output data product, this is recorded in the database and linke to the task. 

Every sub-class of {obj}`astra.base.TaskInstance` takes an optional keyword argument called `input_data_products`. If you don't care so much for data products, you can just provide a filename path to `input_data_products` and `astra` will handle the recording of data products in the database. 


:::{important}
In SDSS-V data products are recorded in the `astra` database when the upstream data reduction pipelines have created a new file. When that happens, `astra` parses the file headers and links that data product to an individual source in the catalog database. That means you can query for an individual source in the catalog, and immediately access all data products that are linked to that source, including their file types (e.g., apStar, apVisit).
:::


## Using pre- and post-execute hooks

Depending on your analysis, you can also define `pre_execute()` and `post_execute()` methods that will be executed before and after the task execution. Examples where you may want to use these could be if you are pre-loading a model for many executions (`pre_execute`), or if you want to write the outputs of your analysis to a database (`post_execute`). You could just put everything in `execute`, but `astra` reports the time taken for `pre_execute`, `execute`, and `post_execute`, so separating your analysis logic can be good for tracking performance. 

## Bundling

Often you want to run the same kind of task on many different data products. Here a data product might be an ApStar spectrum, or some kind of observation. Each of those tasks might have slightly different parameters for each data product (e.g., initial estimate of some parameter), but there are some parameters that will be the same for all tasks. If there is some overhead (CPU or I/O) to executing that task, then it is sensible to bundle tasks together wherever possible to minimise the overhead costs. 

In `Astra` you can bundle tasks together if they share the same *bundled parameters*. A bundled parameter is one that is necessary to load any overheads for a group of tasks. For example, if your task needed to load a model and then estimate the properties of a star given that model, then your parameters might look like:

**Bundled parameters**:
- `model_path`: the path to the model

**Non-bundled parameters**:
- `initial_teff`: initial estimate of the effective temperature
- `initial_logg`: initial estimate of the surface gravity
- `initial_fe_h`: initial estimate of the metallicity

Tasks like this with the same `model_path` can be bundled together so that the model is only loaded once, and executed on many different stars, even if each star has a different `initial_teff`, `initial_logg`, and `initial_fe_h`. In this situation, the time taken to load the model will be automatically inferred and recoreded in the database, as well as the time taken to estimate properties of each star.

Bundled parameters are set in the class definition using the `bundled` argument. The execution logic needs to be a little different when you expect to have some tasks bundled together. Specifically, instead of accessing non-bundled parameter values like `self.initial_teff`, you should use the {func}`astra.base.TaskInstance.iterable` method to iterate over any tasks in the bundle. It's good practice to use the `iterable()` method whether you anticipate bundling tasks together or not.

Here's an example of what it might look like:

```python
import os
from astra.base import TaskInstance, Parameter
from astra.database.astradb import Task, DataProduct

class AnalysisTask(TaskInstance):

    model_path = Parameter(bundled=True)

    initial_teff = Parameter()
    initial_logg = Parameter()
    initial_fe_h = Parameter()

    sigma_clip_limit = Parameter(default=3)    

    def pre_execute(self):
        """ Pre-execute hook called before execution. """

        # Let's explain how self.iterable() works!
        for task, data_products in self.iterable():

            # Let's check the types of these things.

            # Here, task is the database record (a Task object).
            # It's *not* a TaskInstance.
            assert isinstance(task, Task)

            # And data_products is a list of DataProduct objects,
            # even if the user only gave file paths!
            for data_product in data_products:
                assert isinstance(data_product, DataProduct)

                # The path of the data product is given by the 
                # data_product.path attribute
                assert os.path.exists(data_product.path)

            # All the parameters for this task are given by
            # the task.parameters attribute, which includes
            # any default parameters.
            for key, parameter in task.parameters.items():
                print(key, value)

        return None


    def execute(self):

        # Since model_path is a bundled parameter, we can access it like this.
        # (For non-bundled parameters, use the iterable()!)
        model = load_model(self.model_path)

        results = []
        for task, data_products in self.iterable():

            task_results = []
            for data_product in data_products:
                data = load_data_product(data_product.path)

                result = model.estimate_stellar_properties(
                    task.parameters["initial_teff"],
                    task.parameters["initial_logg"],
                    task.parameters["initial_fe_h"],
                )
                task_results.append(result)

            # Finished with this task in the bundle
            results.append(task_results)
        
        return results

```

In this example we use the `pre_execute` hook just to show how {func}`astra.base.TaskInstance.iterable` works. Internally, `iterable()` is used to time how long the analysis takes for a task within a bundle. So if we created a task bundle with 10 tasks in it, the time it took to go through each item in `iterable()` is recorded as the time that task took to complete. The time taken to load the model (the overhead for the bundle) is calculated by taking the total time to complete `execute()` and subtracting the total time recorded for each task from `iterable()`. All of these times are recorded in the database.


### Creating bundles

Now we can create a task bundle in a few different ways:

```python
# Give different values for a couple of un-bundled parameters.
# In this case, astra will use initial_fe_h = 0 for both data products.
A = AnalysisTask(
    input_data_products=["apStar-1.fits", "apStar-2.fits"],
    model_path="my_model.pkl",
    initial_teff=(5000, 5125),
    initial_logg=(3.5, 3.75),
    initial_fe_h=0
)

# Here we will use the same un-bundled parameter values for 
# all data products.
B = AnalysisTask(
    input_data_products=["apStar-1.fits", "apStar-2.fits"],
    model_path="my_model.pkl",
    initial_teff=5777,
    initial_logg=4.4,
    initial_fe_h=0
)

# Or, be explicit
C = AnalysisTask(
    input_data_products=["apStar-1.fits", "apStar-2.fits"],
    model_path="my_model.pkl",
    initial_teff=(5125, 5000),
    initial_logg=(4.4, 2.5),
    initial_fe_h=(0, 0.1)
)
```

The model will only be loaded once when these tasks are executed. If you have a lot of tasks to create but you don't want to figure out the bundling yourself, you can have `astra` do it for you. In the example below we have 10 models and 100 observations. Each observation will be analysed by a random number of models. 

```python
import numpy as np
from astra.base import bundler

# Create some random model paths.
model_paths = [f"my_model_{i}.pkl" for i in range(10)]

# 100 observations
observation_paths = [f"apStar-{i}.fits" for i in range(100)]

M = len(model_paths)

tasks = []
for observation_path in observation_paths:
    # Chose a random number of models to use
    N_models = np.random.choice(1, M)

    # Chose N_models random models
    for model_path in np.random.choice(model_paths, size=M, replace=True):
        task = AnalysisTask(
            input_data_products=observation_path,
            model_path=model_path,
            initial_teff=np.random.uniform(4000, 6000),
            initial_logg=np.random.uniform(0, 5),
            initial_fe_h=np.random.uniform(-1.5, 0)
        )
        tasks.append(task)

# Bundle tasks together if they share all the same bundled parameters to minimize overhead.
for instance in bundler(tasks):
    instance.execute()
```

In the database every `AnalysisTask` we created will have it's own record, with performance information. The bundle just describes the context in which those tasks were executed. 

## Database


## Parameter types

Most parameters are not strictly typed because the parameters of executed tasks are serialized and stored in a database. 

The only parameters that are typed are {obj}`astra.base.DictParameter` and {obj}`astra.base.TupleParameter`, because the lengths of these parameter types are not considered when bundling tasks.


## one data product per task? many per task? 

## where things live (in contrib.[NAME].base, etc)
## where database models live

