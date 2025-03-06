Neural Interactive Proofs Documentation
=======================================

.. rst-class:: center

   `arXiv <https://arxiv.org/abs/2412.08897>`__ |
   `OpenReview <https://openreview.net/forum?id=R2834dhBlo>`__ |
   `GitHub <https://github.com/SamAdamDay/neural-interactive-proofs>`__ |
   `Website <https://neural-interactive-proofs.com>`__

This is the documentation of the experiments for the ICLR 2025 paper "`Neural
Interactive Proofs <https://arxiv.org/abs/2412.08897>`__" :cite:`Hammond2018`. The source
code is available on GitHub at `SamAdamDay/neural-interactive-proofs
<https://github.com/SamAdamDay/neural-interactive-proofs>`__.

The codebase is designed to be easy to use and extend. The :doc:`guides <guides/index>`
section contains tutorials on how to use the codebase, and the :doc:`reference
<reference/index>` section contains a comprehensive API reference.


Quickstart
----------

To get experiments running fast, follow the steps below:

1. Install the package by following the 
   :doc:`installation instructions <guides/installation>`. Alternatively, build and run 
   a Docker container by following the :doc:`Docker instructions <guides/docker>`.
2. Create an experiment script by following the 
   :doc:`guide to running experiments <guides/running_experiments>`. A basic example is:

   .. code-block:: python

      from nip import HyperParameters, run_experiment

      hyper_params = HyperParameters(
         scenario="code_validation",
         trainer="pure_text_ei",
         dataset="lrhammond/buggy-apps",
         interaction_protocol="nip",
      )

      run_experiment(hyper_params)


Contents
--------

.. toctree::
   :maxdepth: 2
   
   guides/index
   reference/index
   changelog
   glossary
   bibliography


Citation
--------

.. code-block:: bibtex

   @inproceedings{neural_interactive_proofs,
      author = {Lewis Hammond and Sam Adam-Day},
      title = {Neural Interactive Proofs},
      booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
      year = {2025},
      eprint={2412.08897},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
   }
