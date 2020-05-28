
.. _astra-tutorials:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Using The Payne with APOGEE spectra

:tocdepth: 3

Using The Payne with APOGEE spectra
===================================

In this tutorial you will use *The Payne* to estimate stellar parameters for a star observed with the APOGEE spectrograph.

*The Payne* is incorporated in Astra as an analysis component. Astra uses `<https://luigi.readthedocs.io/en/stable/>`_ to manage tasks and their dependencies. You can `read more about tasks here <../tasks.html>`_, but in summary we will define some tasks that will::

1. Continuum normalise spectra from individual visits, and combine them.

2. Train *The Payne* using a pre-computed grid of model spectra.

3. Execute *The Payne* using the trained model.

You can see that Task 3 depends on Task 1 and Task 2 being completed. Here we will just explicitly write that into code and let Astra manage when those tasks are executed. 


Installation
~~~~~~~~~~~~

This workflow assumes that you have `installed Astra <../install.html>`_, run the initial setup command, and that you have added the FERRE component::

  # Set up Astra
  astra setup

  # Install the Astra component code for The Payne
  astra component add astra_thepayne 


Continuum normalisation
~~~~~~~~~~~~~~~~~~~~~~~

There are many ways to perform continuum normalisation, and each method will have different input parameters. Astra contains many utility functions so that you can chose what method of continuum normalisation to perform. That means you can easily specify a new workflow just by specifying a different task. Here is what our continuum normalisation task might look like::

  from astra.tasks.io import ApStarFile
  from astra.tasks.continuum import Sinusoidal

  class ContinuumNormalize(Sinusoidal, ApStarFile):

      sum_axis = 0

      def requires(self):
          return ApStarFile(**self.get_common_param_kwargs(ApStarFile))

Here we are creating a class called `ContinuumNormalize` that extends the behaviour of the `ApStarFile` and `Sinusoidal` classes. The `Sinusoidal` and `ApStarFile` classes already define some of the other things we need for this task (like the parameters needed and the `run()` and `output()` functions). The `ApStarFile` class defines that we will be performing work on a file that follows the `apStar data model <https://data.sdss.org/datamodel/files/>`_, and the `Sinusoidal` class describes the continuum normalisation method (a combination of sine and cosine functions). We need to include both of these classes because the `Sinusoidal` class could be executed with any kind of spectra.

The `Sinusoidal` class has four parameters that we need to define:

- A required `continuum_regions_path` parameter that describes which pixels are considered continuum.
- A `L` (length) parameter that defaults to 1400.
- A `order` parameter that defaults to 3.
- A `sum_axis` parameter that defaults to `None`. If `None`, then the spectrum will not be stacked. If not `None`, it will stack multiple visits along the given axis.

The `Sinusoidal` class already has an `output()` function defined that will save the continuum-normalized spectrum to the same directory as the input path, but you can overwrite this if you want by defining your own `output()` function.


Training *The Payne*
~~~~~~~~~~~~~~~~~~~~

We already have a task defined to train *The Payne*, called `astra_thepayne.tasks.Train` which takes the following parameters:

- `training_set_path`: the path containing the labelled set (spectra and labels) to use for training and validation
- `n_steps`: the number of steps to train the neural network (default: 100000)
- `n_neurons`: the number of neurons to use in this (single-layer) neural network (default: 300)
- `weight_decay`: the decay to apply to the weights (default: 0)
- `learning_rate`: the learning rate to apply during training (default: 0.001)

You can `download the training set file used in this workflow (55 MB) <https://drive.google.com/file/d/1RfhkyZBKY3he6sTSM67KPQfVfMnIg_cs/view?usp=sharing>`_, or create one yourself.

Since we don't need to make any changes to the existing `astra_thepayne.tasks.Train` task defined in Astra, let's move straight on to estimating stellar parameters.


Testing *The Payne*
~~~~~~~~~~~~~~~~~~~

Let's define our task to estimate stellar parameters using a trained model. There is already a task defined to run the "test step" of *The Payne* (see `astra_thepayne.tasks.Test`), but this task could be executed on any kind of spectra. We will need to write a task that extends the behaviour of this class, but recognises that it runs on an apStar file::

    from astra.tasks.io import ApStarFile
    from astra_thepayne.tasks import Train, Test

    class StellarParameters(Test, ApStarFile):

        def requires(self):
            return {
                "model": Train(**self.get_common_param_kwargs(Train)),
                "observation": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
            }

Here it is clear that we require *multiple* tasks to be executed before we can execute the `StellarParameters` task: we need a model to be trained, and we need a continuum-normalised observation.

That's it! Now we are ready to analyse a single star.


Analysing a single star
~~~~~~~~~~~~~~~~~~~~~~~

To do this we will need to execute a task that has all the parameters we need. We haven't talked about it yet, but the `ApStarFile` task requires a bunch of parameters that define the spectrum to analyse. These parameters are defined by the SDSS data model. For example, for an apStar spectrum we need:

- `apred`: the reduction version (e.g., "r12")
- `apstar`: define the class of object (e.g., "stars")
- `telescope`: the telescope observed with (e.g., "apo25m")
- `field`: the field the star was observed in (e.g., "000+14")
- `prefix`: the prefix for the file (e.g., "ap") -- this exists for legacy reasons
- `obj`: the object name (e.g., "2M16505794-2118004")

Having these parameters will uniquely define an apStar file, and tell us where we can find it on SDSS servers. Now that we've introduced those parameters, let's look at our workflow file in full to analyse a single star::

    import luigi
    from astra.tasks.base import BaseTask
    from astra.tasks.io import ApStarFile
    from astra.tasks.continuum import Sinusoidal
    from astra_thepayne.tasks import Train, Test


    class ContinuumNormalize(Sinusoidal, ApStarFile):

        sum_axis = 0 # Stack multiple visits.

        def requires(self):
            return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


    class StellarParameters(Test, ApStarFile):

        def requires(self):
            return {
                "model": Train(**self.get_common_param_kwargs(Train)),
                "observation": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
            }

        
    if __name__ == "__main__":
            
        # Do single star.
        file_params = dict(
            apred="r12",
            apstar="stars",
            telescope="apo25m",
            field="000+14",
            prefix="ap",
            obj="2M16505794-2118004",
            use_remote=True # Download the apStar file if we don't have it.
        )

        additional_params = dict(
            n_steps=1000,
            training_set_path="kurucz_data.pkl"
        )

        params = {**file_params, **additional_params}
        
        task = StellarParameters(**params)

        luigi.build(
            [task],
            local_scheduler=True,
            detailed_summary=True
        )


Remember that to run this successfully you will need the `kurucz_data.pkl` file, or your own set of spectra with labels. 

If all goes well, Astra will recognise that the `StellarParameters` task cannot be run until the observations have been continuum-normalised, and until the model has been trained. So you will see that Astra will perform these tasks first, and then estimate stellar parameters given the model you trained. In future if you re-run this workflow Astra will see that a model is already trained, and only train a new model if any of the `Train()` parameters change (e.g., `training_set_path` or `n_neurons` or `n_steps`).


Analysing many stars
~~~~~~~~~~~~~~~~~~~~

If we wanted to run this pipeline on many stars we would just generate many tasks, where each task specifies the parameters that point to the observed data (and any custom parameters you want to set on a per-object basis). In practice we can watch a folder for reduced data products and create a `StellarParameters()` task for every observation. Astra won't re-run any tasks that have already been executed, unless there is a change to the input parameters (e.g., specifying a different `initial_teff` would trigger the tasks to re-run). Alternatively we could load in a list of schedduled observations and create tasks for every observation, and then Astra will only execute those tasks once the apStar file exists. 
