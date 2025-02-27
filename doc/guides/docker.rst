Using a Docker Image
====================

We provide a Docker file (``Dockerfile``) that can be used to build a Docker image. This
Docker image is intended to be used for running experiments, but can also be used for
development.


Requirements
------------

To build a Docker image, you need to fill in the appropriate values in your ``.env``
file (the one you created from ``.env.template`` in the :ref:`installation_steps`). In
particular, you need the following:

- A GitHub personal access token (PAT). This is used to clone the repository inside the Docker
  container, and also push any changes back to the repository. You can create a PAT by following the instructions in the `GitHub documentation
  <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_.
  Ideally, use a fine-grained one that has (read-write) access only to the contents of
  this repository.
- An SSH public key. This allows you to access the Docker container via SSH.
- A Weights & Biases API key. This is used to log experiment data.
- A name (probably your full name) and email address for signing git commits.

If you want to push the image to `Docker Hub <https://hub.docker.com/>`_, you will also need a Docker Hub account.


Building and Pushing the Image
------------------------------

Once you have filled in the appropriate values in your ``.env`` file, you can build the
Docker image using the following command:

.. code-block:: bash

    docker build -t DOCKER_USER/DOCKER_REPO:DOCKER_TAG --target default --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .

replacing ``DOCKER_USER`` with your Docker Hub username, and ``DOCKER_REPO`` and
``DOCKER_TAG`` suitable Docker repository and tag names (e.g.
"neural-interactive-proofs/default").

Alternatively, you can build an image with all of the datasets already downloaded. This
will result in a much larger image, but can make the process of spinning up and running
a new instance faster overall, if using a large dataset. To do this, use the "datasets"
target as follows:

.. code-block:: bash

    docker build -t DOCKER_USER/DOCKER_REPO:DOCKER_TAG --target datasets --secret id=my_env,src=.env --build-arg CACHE_BUST=`git rev-parse main` .

To push the image to Docker Hub, use the following command:

.. code-block:: bash

    docker push DOCKER_USER/DOCKER_REPO:DOCKER_TAG
