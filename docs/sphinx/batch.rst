
.. title:: Batching tasks

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Batching tasks

:tocdepth: 1

.. rubric:: :header_no_toc:`Batching tasks`

There is usually some computational overhead in analysing a stellar spectrum.
For example, large grids of models may need to be loaded into memory every time a spectrum is analysed.
To minimise this overhead, Astra allows you to execute any task in *batch mode*.

Executing in batch mode
-----------------------

Batch mode means that you can supply information about more than one observation (e.g.,
many `ApStarFile` objects or many `ApVisitFile` objects) and Astra will analyse those
observations together. The way to execute a task in batch mode is just to supply the
`ApStarFile` parameters (or `ApVisitFile`, or similar) as a tuple, instead of a string.
For example::

    # Analyse a single star.
    single_kwds = dict(
        release="dr16",
        apred="r12",
        telescope="apo25m", 
        field="218-04",
        prefix="ap",
        obj="2M06440890-0610126",
        apstar="stars"
    )

    # Analyse two stars in batch mode.
    multiple_kwds = dict(
        release=("dr16", "dr16"),
        apred=("r12", "r12"),
        telescope=("apo25m", "apo25m"),
        field=("218-04", "000+14"),
        prefix=("ap", "ap"),
        obj=("2M06440890-0610126", "2M16505794-2118004"),
        apstar=("stars", "stars")
    )

Note that even if *all* of your observations share the same keyword (e.g., `release` or `apstar`),
the length of the tuple for all `ApStarFile` parameters must be the same.
You should keep other (e.g., non-`ApStarFile`) parameters as they are.
For example, if you were to execute APOGEENet, the other parameters would remain the same 
even if you were analysing 1 spectrum, or 100 spectra::

    from astra.contrib.apogeenet.tasks import EstimateStellarParametersGivenApStarFile

    model_path = "APOGEENet.pt"

    # Estimate stellar parameters for single star.
    task = EstimateStellarParametersGivenApStarFile(
        model_path=model_path, 
        **single_kwds
    )

    # Estimate stellar parameters for two stars.
    task = EstimateStellarParametersGivenApStarFile(
        model_path=model_path, 
        **multiple_kwds
    )



Writing tasks for batch mode
----------------------------

Writing tasks to make use of batch mode is easy. The main things you have to do are to
make sure your task inherits from :py:mod:`astra.tasks.BaseTask`, and to use the
:py:mod:`astra.tasks.BaseTask.get_batch_tasks()` function to get a single-task's worth
of work. 
This function is an iterator that will yield individual tasks, regardless of whether
the task is being run in batch mode or not.
If the task is not being executed in batch mode, then only one task (`self`) will be
yielded. 

Below is an example where the task can be executed in single-mode or batch-mode, without
any extra information being supplied by the user::

    from astra.tasks.base import BaseTask
    from astra.tasks.io import LocalTargetTask

    class MyTask(BaseTask):

        def run(self):

            # Do some expensive operation here that we otherwise would have
            # to do for many tasks (for example, load a model).
            model = self.read_model()        

            for task in self.get_batch_tasks():
                # Read the observation for this individual single task.
                spectrum = task.read_observation()

                # Analyse the star using the model we loaded for all stars.
                result = task.estimate_stellar_parameters(model, spectrum)

                # Write the result of this task.
                task.write_output(result)
            
            return None
        

        def requires(self):
            requirements = dict(model=LocalTargetTask(self.model_path))
            if not self.is_batch_mode:
                requirements.update(
                    observation=ObservedSpectrum(**self.get_common_param_kwargs(ObservedSpectrum))
                )
            return requirements


You can see that you will have to write functions to do some of the expensive work (e.g., `read_model`),
but it is easy to write tasks that can be easily executed in batch mode.
The only potential _gotcha_ is what you need to do in `requires()`.
Here you have to send back different dependencies based on whether the task is running in batch mode or not.
The reasons for this are deep and complex.


Scheduling batch tasks
----------------------

Even if you do not explicitly batch a bunch of stars to be analysed together, Astra may schedule
tasks together to run in batch mode to minimise overhead. 
Let's go through an example to see how this works in practice.

- Let's assume that `Observation` represents an observed spectrum, and you need to supply a `field` and `name` to uniquely identify a single observation::

      spectrum = Observation(field="250+00", name="2M000000+000000")

- Let's assume you a task called `MyAnalysisTask` that runs on `Observation` objects, and you need to supply the parameters `order` and `a` to the `MyAnalysisTask`, as well as the parameters for the `Observation` to analyse::

      task = MyAnalysisTask(a=3, order=5, field="250+00", name="2M000000+000000")

- You need to analyse some stars, but you want to try different values of `order` to see the impact on the results. You create the following tasks and give them to the Astra scheduler::

      individual_tasks = [
          MyAnalysisTask(a=3, order=5, field="250+00", name="2M123456+123456"),
          MyAnalysisTask(a=3, order=10, field="250+00", name="2M123456+123456"),
          MyAnalysisTask(a=3, order=5, field="omegaCen", name="2M003341+289732"),
          MyAnalysisTask(a=3, order=10, field="omegaCen", name="2M003341+289732"),
          MyAnalysisTask(a=-1, order=5, field="250+00", name="2M004562-1234872"),
      ]
    
You have submitted these as individual tasks, but Astra can see that `MyAnalysisTask` is batchable,
and that there are tasks where the non-`Observation` parameters are the same (e.g., these should be batched
together to minimise overhead).
In practice these tasks would be grouped together into just three batch tasks::

    MyAnalysisTask(
        a=3, 
        order=5, 
        field=("250+00", "omegaCen"), 
        name=("2M123456+123456", "2M003341+289732")
    )

    MyAnalysisTask(
        a=3, 
        order=10, 
        field=("250+00", "omegaCen"), 
        name=("2M123456+123456", "2M003341+289732"),
    )

    MyAnalysisTask(a=-1, order=5, field="250+00", name="2M123456+123456")

Even though changing `a=-1` from `a=3` might not have any change on how the analysis is performed,
Astra doesn't know that. 
All it can assume is that if a non-`Observation` parameter is different, then that should be run
in a separate batch. 
That's because Astra isn't smart enough to know that changing `a` is unimportant, but changing 
something like `model_path` is important.
All parameters are assumed to have an effect on the output, unless you specify them to be insignificant
when writing the task class.

