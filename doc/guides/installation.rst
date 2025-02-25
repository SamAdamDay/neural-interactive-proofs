Installation
============

This guide will help you install the necessary software to run the library. The library
has been tested on Linux and MacOS, and may work on Windows as well. We also provide a
:doc:`Docker file <docker>`, which can be used to run the library in a container or for
development.


Prerequisites
-------------

- The library requires Python 3.11 or later. 
- You need `git <https://git-scm.com>`_ to clone the repository.
- To log experiment data, you will need a `Weights & Biases <https://wandb.ai/site>`_
  account.
- To run experiments with OpenAI models, you need an OpenAI API key. You can get one by
  signing up at `OpenAI <https://platform.openai.com>`_. Note that in general the use of
  the OpenAI API is not free.

We recommend using a `virtual environment
<https://docs.python.org/3/library/venv.html>`_ to install the library, to avoid
conflicts with other Python packages.


.. _installation_steps:

Installation Steps
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/SamAdamDay/neural-interactive-proofs.git

2. Change to the repository directory:

   .. code-block:: bash

      cd neural-interactive-proofs

3. Install the requirements:

   .. tabs::
     
      .. code-tab:: bash Just Running Experiments

         pip install wheel
         pip install -r requirements.txt
     
      .. code-tab:: bash Also Development

         pip install wheel
         pip install -r requirements_dev.txt

4. Install the library locally in edit mode:

   .. code-block:: bash

      pip install -e .

5. Log in to Weights & Biases:

   .. code-block:: bash

      wandb login

6. Copy the template secrets file:

   .. code-block:: bash

      cp .env.template .env

   Edit the ``.env`` file and fill in the necessary information for your use case. The
   comments in the file should guide you on what to fill in.


Next Steps
----------

See the :doc:`running_experiments` guide for information on how to run experiments.
