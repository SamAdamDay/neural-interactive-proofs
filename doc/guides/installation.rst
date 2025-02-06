Installation
============

This guide will help you install the necessary software to run the library. The library
has been tested on Linux and MacOS, and should work on Windows as well. We also provide
a Docker file, which can be used to run the library in a container or for development.


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


Installation Steps
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/SamAdamDay/pvg-experiments.git

2. Change to the repository directory:

   .. code-block:: bash

      cd pvg-experiments

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
