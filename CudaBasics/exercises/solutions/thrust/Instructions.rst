Instructions
============

In this exercise we will generate a large number of random numbers and sort them using thrust_, once on the GPU and once on the CPU.

Todo
----

Take a look at the source code in ``ThrustSort.cu`` and fill in the TODOs. Hint: Thrust's version of `std::vector` of data which resides on the *CPU* (/host) side are called ``thrust::host_vector``.

Compile the code using

.. code-block:: bash

   make

Submit your compiled application to JURON's batch system by

.. code-block:: bash

   make run

.. _thrust: http://thrust.github.io/

Experiment with different size of ``WORK_SIZE``, defined at the top of the source file.
