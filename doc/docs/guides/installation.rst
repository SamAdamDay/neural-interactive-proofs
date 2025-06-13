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


.. _installation_steps:

Installation Steps
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/SamAdamDay/neural-interactive-proofs.git

   Alternatively, you may wish to fork the repository and clone your fork.

2. Change to the repository directory:

   .. code-block:: bash

      cd neural-interactive-proofs

3. Install the requirements:

   .. tabs::
     
      .. code-tab:: bash Just Running Experiments

         uv sync --no-dev
     
      .. code-tab:: bash Also Development

         uv sync
     
      .. code-tab:: bash Hosting the Language Model Server

         uv sync --extra lm-server
     
      .. code-tab:: bash Using ``pip``

         pip -m venv .venv
         source .venv/bin/activate
         pip install wheel
         pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
         pip install -e --group dev .

4. Log in to Weights & Biases:

   .. code-block:: bash

      wandb login

5. Copy the template secrets file:

   .. code-block:: bash

      cp .env.template .env

   Edit the ``.env`` file and fill in the necessary information for your use case. The
   comments in the file should guide you on what to fill in.


Next Steps
----------

See the :doc:`running_experiments` guide for information on how to run experiments.
