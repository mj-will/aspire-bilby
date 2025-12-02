Installation
============

``aspire-bilby`` is published on PyPI. Install with:

.. code-block:: bash

    pip install aspire-bilby

For development installs, clone the repository and run:

.. code-block:: bash

    pip install -e .[test]

To enable optional backends (e.g., ``torch`` or ``jax``) install the matching
extra defined in ``pyproject.toml``. We also recommend installing ``minipcn``
for SMC sampling:

.. code-block:: bash

    pip install -e .[torch,minipcn]
    pip install -e .[jax,minipcn]
