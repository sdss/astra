# Parameters

A parameter is a variable that might change the analysis result for some input data product.

For example, imagine that the task description was to fit a line to some data set.
You might want to have a boolean (true or false) parameter to indicate whether you should fit outlier data points as well, or you might include a parameter where we mask any data below that parameter value.
Or the parameter might only change how the task is executed, without changing the output (e.g., number of processes to use).
The expected output will be the same, but we might expect to see a difference in the time taken per task, which is tracked by Astra.

All parameters must be defined in the class definition of a task instance. In general, parameters are not strictly typed because they need to be serialised to and from a database.
However, if you expect a tuple (or list-like) input for one input then you **must** use {obj}`astra.base.TupleParameter`, and similarly if you expect a dictionary for one input then you **must** use a {obj}`astra.base.DictParameter`.
Astra won't enforce type checking at runtime, so your code won't immediately break if you supply a tuple to a parameter that is defined by the generic {obj}`astra.base.Parameter` object instead. However, you will encouter problems when trying to [bundling tasks together](bundles), which we will cover in the next section of this guide.
And if you're using a {obj}`astra.base.TupleParameter` then you should assume it could be given as a list (e.g., use duck-typing instead of assuming it will be a `tuple`).

In summary, there are three parameter types:

- {obj}`astra.base.Parameter`: suitable for most parameter types (integers, floats, strings, `None`, etc)
- {obj}`astra.base.TupleParameter`: required if you expect a tuple (or list-like) as input
- {obj}`astra.base.DictParameter`: required if you expect a dictionary as input

These parameter types work in exactly the same way. For example:

```python
import numpy as np
from astra.base import (TaskInstance, Parameter, TupleParameter)

class PlaceYourBets(TaskInstance):

    """ Shut up and take my money! """

    random_seed = Parameter()
    horse_names = TupleParameter()
    pick_top_n = Parameter(default=1)

    def execute(self):
        np.random.seed(self.random_seed)
        indices = np.random.choice(
            len(self.horse_names),
            size=self.pick_top_n,
            replace=False
        )

        return [self.horse_names[index] for index in indices]
```

Now we can create a task instance and place some bets:
```python
money_maker = PlaceYourBets(
    horse_names=("Flying Jacket", "Andromeda", "Milky Way Mapper", "One Run Pony"),
    pick_top_n=2,
    random_seed=8
)

# Who will the winners be?!
winners = money_maker.execute()
```

In this example we **could** have defined `random_seed = Parameter(default=None)`. The code will work fine, but it means that each time we run the code we can expect a different answer. **This is bad.** Your tasks should be _idempotent_: each time you run it with some parameters you should expect to get the same result. You should always define a parameter if you think there is a (reasonable) chance it could alter the results.


Now it's time to learn about how task instances are [bundled together for efficiency](bundles).
