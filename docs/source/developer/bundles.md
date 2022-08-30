# Bundles

Astra bundles tasks together that share common overheads.

If you're executing many tasks of the same kind, with similar parameters, then often there is some overhead that is common to each task.
This could be reading something from disk (e.g., loading a model), performing a large database query, or something like that.

## Which tasks get bundled?

The way Astra decides whether two tasks can be bundled together is by their parameters. When you define a parameter in the task instance class, you can use the `bundled` keyword to declare that this parameter *only* affects the common overhead. For example, the model parameter might be something like `model_path`. It doesn't matter what the input data products are, all analyses on those input data products will use `model_path`. If it takes a few minutes to load the model and only a few seconds to analyse a data product, then it makes sense to declare the `model_path` as a **bundled** parameter. If there are many tasks with the same name and the same values for *bundled parameters*, then these can be executed together as one bundle.

The parameter values do not all have to be the same, only the bundled parameters need to be the same. If two tasks of the same kind had the same bundled parameters (e.g., the same `model_path` in this instance), then you could bundle those tasks together even if the two tasks were analysing two different input data products, or even if they had the same input data products but had different non-bundled parameters (e.g., something like normalisation keywords, or initial guesses, etc).

Here's an example:
```python
from astra.base import (TaskInstance, Parameter, TupleParameter)
from astra.database.astradb import DataProduct

class StellarParameters(TaskInstance):
    model_path = Parameter(bundled=True)

    initial_guess = TupleParameter()
    minimum_snr = Parameter(default=1)

    def execute(self):
        ...

# Here's some DataProducts we prepared earlier..
sun_observed = DataProduct.get(kwargs=dict(full="sun.fits"))
star_observed = DataProduct.get(kwargs=dict(full="star.fits"))

task1 = StellarParameters(
    input_data_products=[sun_observed],
    model_path="my_model.pkl",
    initial_guess=(5777, 4.0, 0),
)

task2 = StellarParameters(
    input_data_products=[star_observed],
    model_path="my_model.pkl",
    initial_guess=(4000, 5, 0),
    minimum_snr=3
)

task3 = StellarParameters(
    input_data_products=[sun_observed],
    model_path="my_alternative_model.pkl",
    initial_guess=(5777, 4.0, 0)
)
```

In this situation, `task1` and `task2` can be bundled together because they share the same bundled parameters. `task3` needs to be run on it's own.

## How do they get bundled?

Task bundling can happen automatically when you create a task (e.g., with lists of parameters), if you already know which tasks should be bundled together.

Or you can create many tasks and use {obj}`astra.utils.bundler` to work out which tasks can be bundled together. This is the typical case in Astra, where the bundling happens somewhere in a directed acyclic graph (DAG) in Airflow. We'll cover how to do that later.

A task bundle is represented in the Astra database as a {obj}`astra.database.astradb.Bundle`, and the {obj}`astra.database.astradb.TaskBundle` table allows for a many-to-many relationship between tasks and bundles. One bundle might contain thousands of tasks, and a second bundle might contain a subset of those tasks. This can be useful for testing different parts of your code before running it on a larger data set, or for splitting up bundles for maximum compute efficiency.

```{todo}
Provide code example
```

## Creating instances from tasks or bundles

A single task instance can be recreated from the database from a {obj}`astra.database.astradb.Task`, or from a task bundle {obj}`astra.database.astradb.Bundle`.

```python
from astra.database.astradb import Task, Bundle

# Get a task and bundle (any will do)
task = Task.get()
bundle = Bundle.get()

# Turn them into task instances
task_instance = task.instance()
bundle_instance = bundle.instance()

# Execute!
task_instance.exeute()
bundle_instance.execute()
```

In this example, `bundle_instance` is a sub-class of {obj}`astra.base.TaskInstance` that just contains lists of values for parameters (one value per task in the bundle), whereas `task_instance` is a single task. For example:
```python
# Let's print the value of 'my_parameter' for the task.
print(task_instance.my_parameter)
# 3

# Now what about the bundle?
print(bundle_instance.my_parameter)
# [6, 1, 5, 3, 4, 8]
```

That means that if we want our task to be able to be executed both on it's own, and within a bundle, we need to be know how to handle situations where the `.parameter` attributes have different types.

## Writing bundle-able tasks

The {obj}`astra.base.TaskInstance` class (which is the parent class to all your task instance classes) includes some handy functions to make it easier to deal with bundled tasks. The easiest way to write your code so that it can handle single tasks or bundled tasks is to use the {func}`astra.base.TaskInstance.iterable` function. This function will return a generator that yields every task in the bundle, as well as the data products for that task, and their parameters. If your task is not in a bundle, this generator will still behave normally (as if it was a single task bundle), ensuring that the code runs the same way for single or bundled tasks.

Let's define a task instance class where we have some overhead that will be common to all tasks.
Here we will also introduce the `pre_execute` function. This is a function that, when defined, will be executed automatically before `execute`.
We'll use it here to explain the object types, and to separate this explanation away from the actual work to do in `execute`.


```python
import pickle
from astra.tools.spectrum import Spectrum1D
from astra.database.astradb import Task, DataProduct
from astra.base import (TaskInstance, Parameter, DictParameter)

class MyAnalysisTask(TaskInstance):

    """ My example analysis task. """

    model_path = Parameter(bundled=True)

    initial_guess = DictParameter()
    normalisation_order = Parameter()

    def pre_execute(self):
        # Iterate over the tasks (either one task or a bundle)
        # and check that the object types are as we expect.
        for task, input_data_products, parameters in self.iterable():

            # Here, the "task" variable is a database record
            assert isinstance(task, Task)

            # The input_data_products is a list of DataProduct objects for this task
            assert isinstance(input_data_products, list)
            for data_product in input_data_products:
                assert isinstance(data_product, DataProduct)

            # The "parameters" is a dictionary of parameters for this task.
            assert isinstance(parameters, dict)

        return None


    def execute(self):

        # Load the model.
        with open(self.model_path, "rb") as fp:
            model = pickle.load(fp)

        # Iterate over all tasks (either one task or a bundle).
        results = []
        for task, input_data_products, parameters in self.iterable():

            # This task will only ever allow one data product per task,
            # so let's just load the first data product (a spectrum)
            dp, = input_data_products
            spectrum = Spectrum1D.read(dp.path)

            # Apply some normalisation.
            continuum = np.polyfit(
                spectrum.wavelength,
                spectrum.flux,
                parameters["normalisation_order"]
            )

            # Fit the spectrum.
            result = model.fit(
                spectrum.flux / continuum,
                initial=parameters["initial_guess"]
            )
            results.append(result)

        return results

```

Here you can see that the bundled parameter (`model_path`) is only ever used when we load the model. We don't need it for individual tasks in a bundle. And you can see that each task in the bundle can have different values of `normalisation_order`, or `initial_guess`.

```{warning}
I was never really happy with how this part of the API turned out, because it seems nicer to access parameters as attributes instead of dictionaries. There's a chance this part could change in future, but that change would happen in a nice, deprecated way, because too much code already depends on this API.
```

## Timing information

Astra records the time a task takes to complete in the database. There are multiple times that are recorded automatically. These are all accesible as attributes on the {obj}`astra.database.astradb.Task` object, and all times recorded are in units of seconds.

Since the `iterable()` method yields through each task one-by-one, Astra can figure out approximately how long it took to analyse each spectrum by taking the difference in time between when one task is yielded and the next.
If we add up the time taken to go through each item in `iterable()`, and subtract this from the total time taken to run `execute()`, the difference tells us approximately how much time was spent in common overheads (e.g., loading the model).

The full list of times recorded for every task are:

```{list-table}
:header-rows: 1

* - Attribute
  - Description
* - `time_total`
  - Total time taken to complete the task: the sum of `time_pre_execute`, `time_execute`, and `time_post_execute`.
* - `time_pre_execute`
  - Time taken to run the `pre_execute()` function.
* - `time_pre_execute_bundle_overhead`
  - Estimated time spent in `pre_execute()` for common overheads for all tasks in a bundle.
* - `time_pre_execute_task`
  - Estimated time spent in `pre_execute()` for this task: `time_pre_execute` - `time_pre_execute_bundle_overhead`.
* - `time_execute`
  - Time taken to run the `execute()` function.
* - `time_execute_bundle_overhead`
  - Estimated time spent in `execute()` for common overheads for all tasks in a bundle.
* - `time_execute_task`
  - Estimated time spent in `execute()` for this task: `time_execute` - `time_execute_bundle_overhead`.
* - `time_post_execute`
  - Time taken to run the `post_execute()` function.
* - `time_post_execute_bundle_overhead`
  - Estimated time spent in `post_execute()` for common overheads for all tasks in a bundle.
* - `time_post_execute_task`
  - Estimated time spent in `post_execute()` for this task: `time_post_execute` - `time_post_execute_bundle_overhead`.
```

Now that we're getting into the weeds, it's probably time to describe all the [database tables](database).
