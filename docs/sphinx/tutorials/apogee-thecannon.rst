
.. _astra-tutorials:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Using The Cannon with APOGEE spectra

:tocdepth: 3

Using *The Cannon* with APOGEE spectra
===================================

In this tutorial you will use *The Cannon* to estimate stellar parameters for a star observed with the APOGEE spectrograph.

*The Cannon* is incorporated in Astra as an analysis component. Astra uses `<https://luigi.readthedocs.io/en/stable/>`_ to manage tasks and their dependencies. You can `read more about tasks here <../tasks.html>`_, but in summary we will define some tasks that will::

1. Continuum normalise spectra from individual visits, and combine them.

2. Train *The Cannon* using a pre-computed set of spectra.

3. Execute *The Cannon* using the trained model.

You can see that Task 3 depends on Task 1 and Task 2 being completed. Here we will just explicitly write that into code and let Astra manage when those tasks are executed.


Installation
~~~~~~~~~~~~

This workflow assumes that you have `installed Astra <../install.html>`_, run the initial setup command, and that you have added *The Cannon* component::

  # Set up Astra
  astra setup

  # Install the Astra component code for *The Cannon*
  astra component add astra_thecannon


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


Creating a training set
~~~~~~~~~~~~~~~~~~~~~~~

You can `download the training set file used in this workflow (224 MB) <https://drive.google.com/file/d/1Fv5nJkowWxAEAy-OaWUQXe01it1jHZH_/view?usp=sharing>`_, or create one yourself. The example file here includes 1624 red giant spectra and labels from DR14. This will be easier to do in future (e.g., by specifying groups of apStar data models and results), but for now you will have to follow a particular file format if you want to create your own training set file.


The training set file should be a binary `pickle <https://docs.python.org/3/library/pickle.html>`_ file that contains a dictionary with the following keys:

- `wavelength`: an array of shape `(P, )` where `P` is the number of pixels
- `flux`: an array of flux values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
- `ivar`: an array of inverse variance values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
- `labels`: an array of shape `(L, N)` where `L` is the number of labels and `N` is the number observed spectra
- `label_names`: a tuple of length `L` that describes the names of the labels


Training *The Cannon*
~~~~~~~~~~~~~~~~~~~~

We already have a task defined to train *The Cannon*, called `astra_thecannon.tasks.Train` which takes the following parameters:

- `training_set_path`: the path containing the labelled set (spectra and labels) to use for training
- `label_names`: the label names to use for the model (your training set data can include more labels than what you use in your model)
- `order`: the polynomial order to use (default: 2)
- `regularization`: the L1 regularization strength to use (default: 0)

Since we don't need to make any changes to the existing `astra_thecannon.tasks.Train` task defined in Astra, let's move straight on to estimating stellar parameters.


Testing *The Cannon*
~~~~~~~~~~~~~~~~~~~

Let's define our task to estimate stellar parameters using a trained model. There is already a task defined to run the "test step" of *The Cannon* (see `astra_thecannon.tasks.Test`), but this task could be executed on any kind of spectra. We will need to write a task that extends the behaviour of this class, but recognises that it runs on an apStar file::

    from astra.tasks.io import ApStarFile
    from astra_thecannon.tasks import Train, Test

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
    from astra_thecannon.tasks import Train, Test


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
            use_remote=True # Download the remote SDSS file if we don't have it.
        )

        additional_params = dict(
            order=2,
            label_names=("TEFF", "LOGG", "FE_H"),
            training_set_path="dr14-apogee-giant-training-set.pkl",
        )

        params = {**file_params, **additional_params}
        task = StellarParameters(**params)

        luigi.build(
            [task],
            local_scheduler=True,
            detailed_summary=True
        )


Remember that to run this successfully you will need the `dr14-apogee-giant-training-set.pkl` file (`download <https://drive.google.com/file/d/1Fv5nJkowWxAEAy-OaWUQXe01it1jHZH_/view?usp=sharing>`_), or your own set of spectra with labels.

If all goes well, Astra will recognise that the `StellarParameters` task cannot be run until the observations have been continuum-normalised, and until the model has been trained. So you will see that Astra will perform these tasks first, and then estimate stellar parameters given the model you trained. In future if you re-run this workflow Astra will see that a model is already trained, and only train a new model if any of the `Train()` parameters change (e.g., `training_set_path` or `order` or `labels`). Here is what the output looks like for me::

    (astra) io:astra arc$ python workflow_example_tc.py
    DEBUG: Checking if StellarParameters(release=DR16, prefix=ap, obj=2M16505794-2118004, apred=r12, telescope=apo25m, apstar=stars, field=000+14, label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0, N_initialisations=10, use_derivatives=True) is complete
    DEBUG: Checking if Train(label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0) is complete
    DEBUG: Checking if ContinuumNormalize(release=DR16, prefix=ap, obj=2M16505794-2118004, apred=r12, telescope=apo25m, apstar=stars, field=000+14, L=1400, order=2, continuum_regions_path=/Users/arc/research/projects/astra_components/astra_thecannon/python/astra_thecannon/etc/continuum-regions.list) is complete
    INFO: Informed scheduler that task   StellarParameters_10_r12_stars_b3efd3bdab   has status   PENDING
    INFO: Informed scheduler that task   ContinuumNormalize_1400_r12_stars_06947744c6   has status   DONE
    DEBUG: Checking if TrainingSetTarget(training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl) is complete
    INFO: Informed scheduler that task   Train___TEFF____LOGG___2_0_be3efdf318   has status   PENDING
    INFO: Informed scheduler that task   TrainingSetTarget__Users_arc_resea_cb09f040ca   has status   DONE
    INFO: Done scheduling tasks
    INFO: Running Worker with 1 processes
    DEBUG: Asking scheduler for work...
    DEBUG: Pending tasks: 2
    INFO: [pid 28197] Worker Worker(salt=593833008, workers=1, host=io.local, username=arc, pid=28197) running   Train(label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0)
    INFO: Training The Cannon model <astra_thecannon.model.CannonModel of 3 labels with a training set of 1624 stars each with 8575 pixels>
    INFO: Training 3-label CannonModel with 1624 stars and 8575 pixels/star
    [===                                                                                                 ]   3% (258/8575)
    [====================================================================================================] 100% (23s)
    INFO: Writing The Cannon model <astra_thecannon.model.CannonModel of 3 labels trained with a training set of 1624 stars each with 8575 pixels> to disk /Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set-Train___TEFF____LOGG___2_0_be3efdf318.pkl
    INFO: [pid 28197] Worker Worker(salt=593833008, workers=1, host=io.local, username=arc, pid=28197) done      Train(label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0)
    DEBUG: 1 running tasks, waiting for next task to finish
    INFO: Informed scheduler that task   Train___TEFF____LOGG___2_0_be3efdf318   has status   DONE
    DEBUG: Asking scheduler for work...
    DEBUG: Pending tasks: 1
    INFO: [pid 28197] Worker Worker(salt=593833008, workers=1, host=io.local, username=arc, pid=28197) running   StellarParameters(release=DR16, prefix=ap, obj=2M16505794-2118004, apred=r12, telescope=apo25m, apstar=stars, field=000+14, label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0, N_initialisations=10, use_derivatives=True)
    INFO: Running test step on 1 spectra
    [=                                                                                                   ] 100% (0s)
    INFO: Inferred labels: [[3.46238829e-01 2.68048519e+00 4.36294368e+03]]
    INFO: Metadata: ({'fvec': array([-2.70198600e+00, -2.57092485e+00,  7.47612006e+00, ...,
        -3.29431575e+06,  4.42269208e+06,  1.14013214e+06]), 'nfev': 22, 'njev': 14, 'fjac': array([[ 1.35724822e+10, -3.82973307e+06, -4.66779051e+06, ...,
            -1.15147764e-03, -9.79146406e-04, -2.97752067e-04],
        [-2.79254727e+09, -8.28716813e+09, -3.90126551e+06, ...,
            9.30447426e-06,  1.66935917e-04,  4.67861350e-04],
        [-3.40363817e+09,  4.00084454e+09, -4.73847745e+09, ...,
            -4.94972200e-04, -5.27099115e-04, -2.23264453e-04]]), 'ipvt': array([1, 3, 2], dtype=int32), 'qtf': array([ 0.94357481, -3.59550803,  3.96755447]), 'x0': array([4.47768058e-02, 2.73183129e+00, 5.02461950e+03]), 'chi_sq': 4.301564877866089e+18, 'ier': 2, 'mesg': 'The relative error between two consecutive iterates is at most 0.000000', 'model_flux': array([0.90422061, 0.90919087, 0.94100578, ..., 1.02329778, 1.01853437,
        1.0016082 ]), 'method': 'leastsq', 'label_names': ('FE_H', 'LOGG', 'TEFF'), 'best_result_index': 8, 'initial_points': 10, 'derivatives_used': True, 'snr': 1005793518.2398922, 'r_chi_sq': 572778279343021.1, 'ftol': 2.220446049250313e-16, 'xtol': 2.220446049250313e-16, 'gtol': 0.0, 'maxfev': 100000, 'factor': 1.0, 'epsfcn': None},)
        -3.29431575e+06,  4.42269208e+06,  1.14013214e+06]), 'nfev': 22, 'njev': 14, 'fjac': array([[ 1.35724822e+10, -3.82973307e+06, -4.66779051e+06, ...,
            -1.15147764e-03, -9.79146406e-04, -2.97752067e-04],
        [-2.79254727e+09, -8.28716813e+09, -3.90126551e+06, ...,
            9.30447426e-06,  1.66935917e-04,  4.67861350e-04],
        [-3.40363817e+09,  4.00084454e+09, -4.73847745e+09, ...,
            -4.94972200e-04, -5.27099115e-04, -2.23264453e-04]]), 'ipvt': array([1, 3, 2], dtype=int32), 'qtf': array([ 0.94357481, -3.59550803,  3.96755447]), 'x0': array([4.47768058e-02, 2.73183129e+00, 5.02461950e+03]), 'chi_sq': 4.301564877866089e+18, 'ier': 2, 'mesg': 'The relative error between two consecutive iterates is at most 0.000000', 'model_flux': array([0.90422061, 0.90919087, 0.94100578, ..., 1.02329778, 1.01853437,
        1.0016082 ]), 'method': 'leastsq', 'label_names': ('FE_H', 'LOGG', 'TEFF'), 'best_result_index': 8, 'initial_points': 10, 'derivatives_used': True, 'snr': 1005793518.2398922, 'r_chi_sq': 572778279343021.1, 'ftol': 2.220446049250313e-16, 'xtol': 2.220446049250313e-16, 'gtol': 0.0, 'maxfev': 100000, 'factor': 1.0, 'epsfcn': None},)
    INFO: [pid 28197] Worker Worker(salt=593833008, workers=1, host=io.local, username=arc, pid=28197) done      StellarParameters(release=DR16, prefix=ap, obj=2M16505794-2118004, apred=r12, telescope=apo25m, apstar=stars, field=000+14, label_names=('TEFF', 'LOGG', 'FE_H'), order=2, training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl, regularization=0, N_initialisations=10, use_derivatives=True)
    DEBUG: 1 running tasks, waiting for next task to finish
    INFO: Informed scheduler that task   StellarParameters_10_r12_stars_b3efd3bdab   has status   DONE
    DEBUG: Asking scheduler for work...
    DEBUG: Done
    DEBUG: There are no more tasks to run at this time
    INFO: Worker Worker(salt=593833008, workers=1, host=io.local, username=arc, pid=28197) was stopped. Shutting down Keep-Alive thread
    INFO:
    ===== Luigi Execution Summary =====

    Scheduled 4 tasks of which:
    * 2 complete ones were encountered:
        - 1 ContinuumNormalize(...)
        - 1 TrainingSetTarget(training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl)
    * 2 ran successfully:
        - 1 StellarParameters(...)
        - 1 Train(...)

    This progress looks :) because there were no failed tasks or missing dependencies

    ===== Luigi Execution Summary =====

    ===== Luigi Execution Summary =====

    Scheduled 4 tasks of which:
    * 2 complete ones were encountered:
        - 1 ContinuumNormalize(...)
        - 1 TrainingSetTarget(training_set_path=/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl)
    * 2 ran successfully:
        - 1 StellarParameters(...)
        - 1 Train(...)

    This progress looks :) because there were no failed tasks or missing dependencies

    ===== Luigi Execution Summary =====


Here you can see that because we asked Astra (or Luigi, rather) to estimate stellar parameters for this star, it found that the `StellarParameters` task requires a Cannon model to be trained, and it also requires the star's apStar spectra to be continuum-normalised. It then found that the `ContinuumNormalize` task for this star had already been executed (with the exact same parameters) so it didn't need to re-execute that task. It then proceeded to train a Cannon model, and use that model to estimate stellar parameters. All of the dependencies were handled automagically for us!


Analysing many stars
~~~~~~~~~~~~~~~~~~~~

If we wanted to run this pipeline on many stars we would just generate many tasks, where each task specifies the parameters that point to the observed data (and any custom parameters you want to set on a per-object basis). In practice we can watch a folder for reduced data products and create a `StellarParameters()` task for every observation. Astra won't re-run any tasks that have already been executed, unless there is a change to the input parameters (e.g., specifying a different `initial_teff` would trigger the tasks to re-run). Alternatively we could load in a list of schedduled observations and create tasks for every observation, and then Astra will only execute those tasks once the apStar file exists.
