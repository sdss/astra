
.. _astra-tutorials:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Tutorials

:tocdepth: 3

.. rubric:: :header_no_toc:`Tutorials`

Example workflow for APOGEE spectra
-----------------------------------

Astra uses `Luigi <https://luigi.readthedocs.io/en/stable/>`_ to manage individual tasks and dependencies. In this context a task is an individual step in an analysis pipeline. This might be performing continuum normalisation, or stacking spectra from multiple visits, or running FERRE. Each one of these is a task, and tasks can depend on the output of each other. To create a workflow (or a pipeline) we need to define multiple tasks. All you need to do is define what each task requires, and Astra will deal with the scheduling and execution of tasks in the correct order.

In this tutorial we will create a workflow for executing FERRE on APOGEE spectra. The tasks we are going to need are::

1. Continuum normalise spectra from individual visits, and combine them.

2. Execute FERRE on the combined spectrum.


Each task needs four things to be properly defined:

1. The parameters of that task.

2. A `requires()` function that defines what other tasks it depends on.

3. A `run()` function that executes the task.

4. An `output()` function that defines the output of the task (usually a local file).



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

The `Sinusoidal` class has a few parameters that we need to define. There are other ways to check what parameters are required, but here I will just show you code from the `Sinusoidal` class so you can see what parameters it requires::

  class Sinusoidal(BaseTask):

      L = luigi.FloatParameter(default=1400)
      order = luigi.IntParameter(default=3)
      continuum_regions_path = luigi.Parameter()

      sum_axis = luigi.IntParameter(default=None)

      def run(self):

          # ...
      

      def output(self):
          # Put the output path relative to the input path.
          output_path_prefix, ext = os.path.splitext(self.input().path)
          stacked_str = f'stacked-{self.sum_axis}' if self.sum_axis is not None else 'original'
          return luigi.LocalTarget(f"{output_path_prefix}-norm-sinusoidal-{self.L}-{self.order}-{stacked_str}.fits")

Here you can see that there are four parameters:

- A required `continuum_regions_path` parameter that describes which pixels are considered continuum.
- A `L` (length) parameter that defaults to 1400.
- A `order` parameter that defaults to 3.
- A `sum_axis` parameter that defaults to `None`. If `None`, then the spectrum will not be stacked. If not `None`, it will stack multiple visits along the given axis.

You can also see that the `output` function of this task: the task will write a continuum-normalised spectrum to a path in the same directory as the input path (e.g. where the apStar file exists).


Running FERRE
~~~~~~~~~~~~~

FERRE is a component in Astra, and that means it has a FERRE task already defined that you can use. That task defines the parameters required, the `run()` command that executes FERRE, and the `output()` function. All you need to do is define what kind of data it will execute, and what task is required before FERRE can run::

  from astra_ferre.tasks import Ferre

  class StellarParameters(Ferre, ApStarFile):
      
      def requires(self):
          return ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
      
Again you can see our task extends the behaviour from the `Ferre` and `ApStarFile` tasks. All we have to do here is define that our `StellarParameters` task requires the `ContinuumNormalize` task to finish first.


Analysing a single star
~~~~~~~~~~~~~~~~~~~~~~~

Let's analyse a single star. To do that we will need to execute a task that has all the parameters we need. We haven't talked about it yet, but the `ApStarFile` task requires a bunch of parameters that define the spectrum to analyse. These parameters are defined by the SDSS data model. For example, for an apStar spectrum we need:

- `apred`: the reduction version (e.g., "r12")
- `apstar`: define the class of object (e.g., "stars")
- `telescope`: the telescope observed with (e.g., "apo25m")
- `field`: the field the star was observed in (e.g., "000+14")
- `prefix`: the prefix for the file (e.g., "ap") -- this exists for legacy reasons
- `obj`: the object name (e.g., "2M16505794-2118004")

Having these parameters will uniquely define an apStar file, and tell us where we can find it on SDSS servers. Now that we've introduced those parameters, let's look at our workflow file in full to analyse a single star::

  from astra.tasks.io import ApStarFile
  from astra.tasks.continuum import Sinusoidal
  from astra_ferre.tasks import Ferre

  class ContinuumNormalize(Sinusoidal, ApStarFile):

      sum_axis = 0

      def requires(self):
          return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


  class StellarParameters(Ferre, ApStarFile):
      
      def requires(self):
          return ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
      

  if __name__ == "__main__":
          
      # Let's run our workflow on a single star.

      import matplotlib.pyplot as plt

      # Do single star.
      file_params = dict(
          apred="r12",
          apstar="stars",
          telescope="apo25m",
          field="000+14",
          prefix="ap",
          obj="2M16505794-2118004",
      )

      additional_params = dict(
          initial_teff=5000,
          initial_logg=4.0,
          initial_m_h=0,
          initial_alpha_m=0.0,
          initial_n_m=0.0,
          initial_c_m=0.0,
          synthfile_paths="~/sdss/astra_components/data/ferre/asGK_131216_lsfcombo5v6/p6_apsasGK_131216_lsfcombo5v6_w123.hdr"
      )

      params = {**file_params, **additional_params}

      spectrum, result = StellarParameters(**params).run()

      params, params_err, model_flux, meta = result

      fig, ax = plt.subplots()
      ax.plot(spectrum.wavelength, spectrum.flux[0], c='k')
      ax.plot(meta["dispersion"][0], model_flux[0], c='r')

      plt.show()

