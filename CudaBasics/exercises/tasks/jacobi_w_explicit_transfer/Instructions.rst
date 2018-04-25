Jacobi with explicit memory allocation and transfers
====================================================
In this exercise, we take the Jacobi example that uses unified memory and change it to use separate
memory allocation on host and device and explicit transfers. The corresponding parts of the code are
marked with todos.

Compile the code with

.. code-block:: bash
	
	make

To run it (using the batch system) call

.. code-block:: bash

	make run

Extra credit: Once you changed the program, generate a profile and examine it with nvvp. How does it 
compare to the profile for the unified memory version.


