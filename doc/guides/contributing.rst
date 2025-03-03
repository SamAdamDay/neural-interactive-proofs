Contributing
============

We welcome contributions to the Neural Interactive Proofs codebase! This document
provides a guide to contributing to the project.


Issues and Pull Requests
------------------------

If you find a bug or have a feature request, please open an issue on GitHub. If you
would like to contribute a small fix (less than 30 minutes work), feel free to open a
pull request directly. For larger changes, please open an issue first to discuss the
change.

Pull requests require at least one review before they can be merged.

We use `GitHub Actions <https://docs.github.com/en/actions>`_ for continuous
integration. When you open a pull request, the tests will be run automatically, and must
pass before the pull request can be merged.


Style Guide
-----------

- The code is formatted using `black <https://black.readthedocs.io/en/stable/>`_.
- The line length is 88 (see `rationale
  <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length>`_).
- Docstrings are formatted using `Numpydoc style
  https://numpydoc.readthedocs.io/en/latest/format.html`_.
- Please add type hints to all functions.


Testing
-------

We use `pytest <https://docs.pytest.org/en/stable/>`_ for testing. To run the tests, use
the following command:

.. code-block:: bash

   python -m pytest

When developing a new feature, please add tests for the new functionality.


Documentation
-------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate the documentation.
To build the documentation, run the following command:

.. code-block:: bash

   sphinx-build doc doc/_build/ -j auto

New features should be documented in the appropriate place in the documentation, located
in the ``doc`` directory.


Pre-commit Hooks
----------------

We use `pre-commit <https://pre-commit.com/>`_ to run checks on the code before
committing. To install the pre-commit hooks, run the following command:

.. code-block:: bash

   pre-commit install


Development Workflow
--------------------

1. Make sure you have submitted an issue on GitHub first if the change is significant
   (will take more than 30 minutes to implement). 

2. Follow the :doc:`instructions to install the library <installation>`_, making sure
   you do the following:

   - Fork the repository on GitHub before cloning.
   - Install the development requirements (``requirements_dev.txt``).

3. Create a new branch for your feature or bug fix:

   .. code-block:: bash

      git checkout -b my_feature

4. Make your changes, add tests for the new functionality and update the documentation.
   Consult the guides and API reference for more information on how to do this.

5. Run the tests:

   .. code-block:: bash

      python -m pytest

   If some tests fail, fix the issues before proceeding.

6. Format with black:

   .. code-block:: bash

      black .

7. Check for linting errors (if you have pre-commit hooks installed, this will be done
   automatically):

   .. code-block:: bash

      ruff check .

8. Check that the documentation builds without errors, and looks correct:

   .. code-block:: bash

      sphinx-build doc doc/_build/ -j auto

   If there are errors, fix them before proceeding.

9. Commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "My feature"

10. Push your changes to your fork:

    .. code-block:: bash

       git push origin my_feature

11. Create a pull request on GitHub. Make sure to include a description of the changes
    and any relevant context.
