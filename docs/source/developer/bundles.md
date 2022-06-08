# Bundles

Astra bundles tasks together that share common overheads.

If you're executing many tasks of the same kind, with similar parameters, then often there is some overhead that is common to each task.
This could be reading something from disk (e.g., loading a model), performing a large database query, or something like that.
The way Astra decides whether two tasks can be bundled together is by their parameters. 

If a parameter is only used for the overheads (e.g., `model_path`) then in the task instance definition we can set this as a **bundled** parameter. 

Here's an example:
```python
from astra.base import (TaskInstance, Parameter, TupleParameter)

class StellarParameters(TaskInstance):
    model_path = Parameter(bundled=True)

    initial_guess = TupleParameter()
    minimum_snr = Parameter(default=1)

    def execute(self):
        ...


task1 = ExampleTask(
    model_path
    A=1, B="hello", C=3)
task2 = ExampleTask(A=5, B="hello", C=10)
task3 = ExampleTask(A=5, B="there", C=10)

```