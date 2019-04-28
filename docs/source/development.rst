Development
===========

Development is best done running flowty inside of docker as this enables testing in its
target environment. It also mitigates the need of setting up all the dependencies.


Using docker
------------
The easiest way to hack on flowty is by mounting the flowty repository over the
installed version in ``/src`` in the application container. e.g.

.. code-block:: console

    $ docker run -it --entrypoint bash \
        --runtime=nvidia \
        --mount type=bind,source=$PWD,target=/src \
        willprice/flowty

One has to be careful though as flowty is installed into the global environment, and the
CLI application to ``/usr/local/bin/flowty``, so once in the bash shell, run the
following:

.. code-block:: console

    $ python3 -m pip uninstall --yes flowty
    $ python3 -m pip install -e .[test]
    $ pytest

The second ``pip`` command will install ``flowty`` in editable mode, that is, it will
link
directly to ``/src`` so when you make changes to any files, when you invoke the tests,
they will run against the updated files.
