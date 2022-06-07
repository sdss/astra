# Tasks

A task represents a unit of work. 

Normally one task will do some work on one data product, but this is not a rule.


## A task instance

A **task instance** is an Python class that Astra knows how to manage, and has some executable code.
All task instances must be sub-classed from {obj}`astra.base.TaskInstance`, and the minimum requirement is that they have an `execute()` function where the work actually happens.


A task instance will perform some analysis on data products, it might write results to a database, and/or it might create some output data products.
Every {obj}`astra.base.TaskInstance` (or subclass thereof) has some parameter(s) that you define in the class definition.
A parameter is a variable that could change the result (for any input data), or could change how it is executed.
A parameter **should not** be something like `input_spectrum_path`, or anything describing the input data. We will describe how data are fed to the task later in this guide. 
For now you should just assume that the task instance somehow _'knows'_ what the input data file is and how to load it.

Let's define a simple task that does not require any data:

```python
from astra.base import TaskInstance, Parameter

class GreedyPrimes(TaskInstance):

    """ Use a greedy approach to return a list of prime numbers. """

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

The `GreedyPrimes` task has only two parameters: `min_value` and `max_value`. The `min_value` parameter has a default value (2), so we only require a `max_value` to create an instance of `GreedyPrimes`. If the task definition requires a parameter, and we try to create a task instance without giving the required parameter(s), a `TypeError` will be raised.

Let's create a task instance by setting `max_value` to 30 and see the result:

``````{tab} Python
```python
ti = GreedyPrimes(max_value=30)
ti.execute()
```
``````
``````{tab} Output
```bash
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```
``````


## Why bother?

It looks like we just wrapped a function in a class for no reason. When we ran `GreedyPrimes.execute()` there were a few things that happened in the background. Specifically:

- A record of the task (`GreedyPrimes`) was recorded in the Astra database (as a {obj}`astra.database.astradb.Task`). 
- The task parameters were also recorded:
    + `min_value`: 2 (default)
    + `max_value`: 30
- The version of `astra` was recorded with the task, as well as some other metadata for reproducibility and provenance.
- The time taken to perform different parts of `GreedyPrimes` is recorded in the database, for tracking performance.

Astra records how long each task takes to execute, and estimates how much of that time was spent in overhead before doing the analysis (e.g., loading a model)  and after the analysis (e.g., writing outputs). 
This lets us track performance with code changes, and understand where time should be spent improving things. If your analysis task writes to the database, or creates data products, then these outputs are linked in the database to the task itself. That means for every analysis output, you know exactly what the parameters were that went into that analysis, and other performance metadata.

This is done quietly in the background for every task instance. And any task instance recorded in the database can be re-constructed later on. For example, let's say you ran analysis pipeline that took a long time, with many complex tasks being executed one after another. Some were executed on a GPU, some were executed on a high-performance computing cluster through a queue system, and some minor tasks were executed on the head node. You can see that the pipeline failed, and which task caused the failure. Instead of re-running the big pipeline with different inputs, verbose bugging, or something like that, you can open an interactive Python terminal and create that task instance from the database:

```python
from astra.database.astradb import Task

failed_task = Task.get(666) # database record of my suspicious task

ti = failed_task.instance() # create an instance of this task

# execute the task interactively with some additional debug keywords
ti.execute(debug=True, verbose=True)
```

When the task fails (again), the interactive debugger will drop you exactly into the code where things fell over. You can inspect variables, test things, and navigate through the stack trace. That means you can quickly discover exactly what went wrong, without having to re-execute any other part of the code, or submit jobs to compute clusters or GPUs, etc. Once you've fixed what went wrong, you can simply resume your pipeline from the point it failed.


```{important}
There are different meanings of the word 'task' that depend on context. 
For example, an 'task instance' is a bit of Python code that can do some work and produce an output (e.g., `GreedyPrimes`). 
Depending on the context, a 'task' might refer to a record in the database (e.g., {obj}`astra.database.astradb.Task`) of some work that was done (by the instance of a task). You can see this in the example above, where we rerieve a task stored in the database (`failed_task`) with the identifier 666, and then use that database record to construct a task instance (`ti`), which we can execute.

Things get more complicated if you're also using [Airflow](airflow-index). In Airflow, a 'task' and 'task instance' refer to bits of work to be executed by Airflow (not necessarily by Astra). This overloading is unfortunate, but we'll give context for what we mean by 'task'. You only have to be aware of it, and try not to think about it.
```

The 'uniqueness' of a task instance is defined by the parameters provided, the input data products given to it, and the version of the code used to execute the task instance. Let's see [how more complex sets of parameters can be defined](parameters).