
.. title:: Components

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Components

:tocdepth: 1

.. rubric:: :header_no_toc:`Components`

Astra includes many external analysis methods as contributed components.
Many of these components include bespoke analysis methods for specialised types of stars.
Below you can find a summary of what components are currently available to run on 
APOGEE or BOSS spectra. In all cases a component *could* be executed on both APOGEE or
BOSS spectra, but we do not have the current models (e.g., spectral grids) to do so.

.. list-table::
    :widths: 50 25 25
    :header-rows: 1

    * - Component
      - APOGEE
      - BOSS
    * - `APOGEENet`_
      - YES
      - NO
    * - `Classifier`_
      - YES
      - YES
    * - `FERRE`_
      - YES
      - NO
    * - `Hot star code`_
      - YES
      - NO
    * - `The Cannon`_
      - YES
      - YES
    * - `The Payne`_
      - YES
      - NO
    * - `WD code`_
      - NO
      - YES

If you are interested in the current functionality of these components, or components
that are planned for integration into Astra, see the `roadmap <roadmap.html>`_.


APOGEENet
=========

**Contributor:** Marina Kounkel (University of Michigan) and collaborators.

APOGEENet uses a neural network to estimate stellar properties of young stellar objects
observed with the APOGEE instrument.
At the time of writing, only the pre-trained neural network for APOGEENet is available.
That means that there are no Astra tasks to train a new neural network.


The most relevant tasks for APOGEENet in Astra are:

- :py:mod:`astra.contrib.apogeenet.tasks.EstimateStellarParametersGivenApStarFile`
- :py:mod:`astra.contrib.apogeenet.tasks.EstimateStellarParameters`

`EstimateStellarParametersGivenApStarFile` will estimate stellar parameters given some
APOGEENet model and an `ApStarFile` object. 
The `EstimateStellarParametersGivenApStarFile` class is a sub-class of the 
`EstimateStellarParameters` class (see below), which is a base task that does not specify what kind
of APOGEE product to expect.


.. inheritance-diagram:: astra.contrib.apogeenet.tasks.EstimateStellarParametersGivenApStarFile
    :top-classes: astra.tasks.base.BaseTask
    :caption: Inheritance diagram for `EstimateStellarParametersGivenApStarFile`.


The only required parameter for `EstimateStellarParameters` is `model_path`: the location
of a file that has the neural network coefficients stored.
The `EstimateStellarParametersGivenApStarFile` task requires the `model_path` parameter,
and any parameters required by `ApStarFile`.

The `EstimateStellarParametersGivenApStarFile` task is `batchable <batch.html>`_: you can analyse many APOGEE observations at once,
minimising the computational overhead in loading the model. 

**Insert APOGEENet workflow example here**


Classifier
==========

**Contributor:** Gabriella Contardo (Flatiron Institute)

This component uses a deep convolutional neural network with drop-out to classify sources
by their spectral type. 
Training sets are available for APOGEE/apVisit and BOSS/spec spectra.



FERRE
=====

**Contributors:** Carlos Allende-Prieto (Instituto de Astrofisica de Canarias), Jon Holtzman (New Mexico State University), and others

FERRE is a code to interpolate pre-computed grids of model spectra and compare with
observations.
The best-fitting model spectrum by chi-squared minimisation, with a few optimisation
algorithms available.
FERRE was used (as part of ASPCAP) for the APOGEE analysis of SDSS-IV data. 
Astra has tasks that reproduce the functionality of ASPCAP.

API
---

.. toctree: api/astra/contrib/ferre/index
    :maxdepth: 2


Hot star code
=============
**Contributors:** Ilya Straumit (KU Leuven)

The Cannon
==========

**Contributors:** Melissa Ness (Columbia University; Flatiron Institute), Andy Casey (Monash), and others

The Cannon :cite:`2015ApJ16N` is a data-driven method to estimate stellar labels 
(e.g., effective temperature, surface gravity, and chemical abundances).
A training set of stars with high-fidelity labels is required to train a model
to predict stellar spectra. 

If you want to use The Cannon as a task in Astra then the most relevant classes are:

- :py:mod:`astra.contrib.thecannon.tasks.train.TrainTheCannon`
- :py:mod:`astra.contrib.thecannon.tasks.test.TestTheCannon`

If you want to use The Cannon without Astra then the most relevant class is:

- :py:mod:`astra.contrib.thecannon.CannonModel`


Train The Cannon using a pre-prepared training set
--------------------------------------------------

You can train The Cannon in Astra using a `pickle` file that contains the training set.
The training set file should contain a dictionary with the following entries:    
    - `wavelength`: an array of shape `(P, )` where `P` is the number of pixels
    - `flux`: an array of flux values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
    - `ivar`: an array of inverse variance values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
    - `labels`: an array of shape `(L, N)` where `L` is the number of labels and `N` is the number observed spectra
    - `label_names`: a tuple of length `L` that describes the names of the labels

Once you have created this file you can supply the path of the training set to the
:py:mod:`astra.contrib.thecannon.tasks.train.TrainTheCannon` task.


Train The Cannon using SDSS spectra and labels
----------------------------------------------

See the workflow file.


.. inheritance-diagram::  astra.contrib.thecannon.tasks.train.TrainTheCannon
    :top-classes: astra.tasks.base.BaseTask
    :parts: 2


Testing The Cannon
------------------

You can estimate stellar labels given some spectra and a trained model using the
:py:mod:`astra.contrib.thecannon.tasks.test.TestTheCannon` task. However, this task
has no hard-coded information about what kind of observation to expect (e.g., APOGEE
or BOSS). That means you need to sub-class this task and inherit the behaviour from
the kind of spectra you would like to use The Cannon on.

For example, if you wanted to train The Cannon on APOGEE apVisit specra, you would
sub-class the `TestTheCannon` task like this::

    import astra
    from astra.tasks.io import ApStarFile
    from astra.contrib.thecannon.tasks.train import TrainTheCannon
    from astra.contrib.thecannon.tasks.test import TestTheCannon

    @astra.inherits(TrainTheCannon, ApStarFile)
    class StellarParameters(TestTheCannon):

        """
        A task to estimate stellar parameters, given an ApStar file and The Cannon.
        """

        def requires(self):
            return {
                "model": TrainTheCannon(**self.get_common_param_kwargs(TrainTheCannon)),
                "observation": ApStarFile(**self.get_common_param_kwargs(ApStarFile))
            }

Now our `StellarParameters` task will know how to load ApStar spectra.



API
---

.. toctree:: api/astra/contrib/thecannon/index
   :maxdepth: 2
   :titlesonly:

.. toctree:: api/astra/contrib/thecannon/tasks/index
   :maxdepth: 2
   :titlesonly:


The Payne
=========

**Contributors:** Yuan-Sen Ting (Australian National University)

The Payne uses a single-layer neural network trained on model spectra to estimate
stellar properties.


WD code
=======

**Contributors:** Nicola Gentile Fusillo (European Southern Observatory)


.. bibliography:: refs.bib
   :style: unsrt