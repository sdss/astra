
.. title:: Database

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Database

:tocdepth: 1

.. rubric:: :header_no_toc:`Database`


The primary Astra database lives in the existing SDSS-V PostgreSQL database server that is 
hosted at Utah. We use the general utilities and functionality for connecting to SDSS databases 
from `sdssdb`, which use object-relational mapping (ORM).

If you want to run Astra locally or direct the database connection to somewhere else, you can do this by changing the settings in your `astra` profile in your `sdssdb` configuration file. The database that Astra connects to will need to have the Astra database schema, which is stored in `schema/astradb/astradb.sql` and can be loaded into PostgreSQL with the terminal command ::

    psql < schema/astradb/astradb.sql

An overview on tasks
--------------------

Analysis work in Astra is organised into individual tasks. Briefly, a task is uniquely defined by the name of the task (e.g., `EstimateSNR` for some ficticious task to estimate a signal-to-noise ratio) and the parameters given to that task. The parameters include information about what file(s) to use to estimate the signal-to-noise ratio. If we create two tasks with the same parameters and execute one task, then when you try to execute the second task Astra will realise that the task has already been executed because it has outputs produced from the first task we ran.


Schema
======

The most relevant tables in the Astra database are:

- `astra.task`: information about individual tasks
- `astra.parameter`: a key-value table to record parameter-value pairs provided to tasks
- `astra.task_parameter`: a junction table to allow many-to-many relationships between tasks and parameters
- `astra.output_interface`: a reference table storing keys of database outputs from all tasks

There are many tables that store results from contributed analysis tasks (e.g., The Cannon). These include:

- `astra.doppler`: radial velocity measurements
- `astra.apogeenet`: stellar parameter estimates from a neural network trained on young stars
- `astra.ferre`: stellar parameters and abundances estimated using FERRE
- `astra.thecannon`: stellar labels (parameters and chemical abundances) estimated using The Cannon
- `astra.thepayne`: stellar labels (parameters and chemical abundances) estimated using a neural network


The `task` table
^^^^^^^^^^^^^^^^

The `task` table stores high-level information about a task. This includes the Python module where the task is defined, a (nearly) unique hash that encodes the task parameters, the status of the task, and other miscellaneous information. Each task is assigned a primary key (`pk`), which is used to reference other tables.

The primary key `pk` is a unique identifier for this table.

+---------------+---------------------+------------------------------------------------------------+
| Column name   | Type                | Description                                                |
+===============+=====================+============================================================+
| `pk`          | serial primary key  | A primary key that defines a reference for the task.       |
+---------------+---------------------+------------------------------------------------------------+
| `task_module` | text                | The Python module where the task is defined.               |
+---------------+---------------------+------------------------------------------------------------+
| `task_id`     | text                | A (nearly) unique hash of a dictionary that contains the   |
|               |                     | task parameter names as keys, and their values.            |
+---------------+---------------------+------------------------------------------------------------+
| `status_code` | int                 | An integer representing the status of the task.            |
+---------------+---------------------+------------------------------------------------------------+
| `duration`    | real                | The time taken for the task to complete, in seconds.       |
+---------------+---------------------+------------------------------------------------------------+
| `created`     | timestamp           | When the task was created in the database.                 |
+---------------+---------------------+------------------------------------------------------------+
| `modified`    | timestamp           | When the task was last updated in the database.            |
+---------------+---------------------+------------------------------------------------------------+
| `output_pk`   | bigint              | A foreign key referencing the primary key of the output    |
|               |                     | interface table (`astra.output_interface(pk)`)             |
+---------------+---------------------+------------------------------------------------------------+


The `parameter` table
^^^^^^^^^^^^^^^^^^^^^

The `parameter` table stores unique key-value pairs of parameters. The primary key `pk` is a unique identifier for this table. The table has a constraint requiring the (`parameter_name`, `parameter_value`) pairs to be unique.

+-------------------+---------------------+------------------------------------------------------------+
| Column name       | Type                | Description                                                |
+===================+=====================+============================================================+
| `pk`              | serial primary key  | A primary key that uniquely references this key-value pair.|
+-------------------+---------------------+------------------------------------------------------------+
| `parameter_name`  | text                | The name of the parameter, as defined in the task.         |
+-------------------+---------------------+------------------------------------------------------------+
| `parameter_value` | text                | The value of the parameter.                                |
+-------------------+---------------------+------------------------------------------------------------+



The `task_parameter` table
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because two tasks can share the same parameter name and value, we use a junction table to store information about which tasks have what parameters. The schema for the junction table looks like this:

+-------------------+---------------------+-------------------------------------------------------------------+
| Column name       | Type                | Description                                                       |
+===================+=====================+===================================================================+
| `pk`              | serial primary key  | A primary key that uniquely references this row.                  |
+-------------------+---------------------+-------------------------------------------------------------------+
| `task_pk`         | bigint              | A foreign key referencing the primary key of the task table.      |
+-------------------+---------------------+-------------------------------------------------------------------+
| `parameter_pk`    | bigint              | A foreign key referencing the primary key of the parameter table. |
+-------------------+---------------------+-------------------------------------------------------------------+



A code example
--------------

Code example

- Define it in the SQL schema. Must have an `output_pk` column as an `int primary key`, which references
`astra.output_interface(pk)`.

- Define the interface in the astradb.py

- Define the output of the task.

- Write it.

- Show that the task durations match up based on what we expect.




Batching tasks
--------------

- Show an example of a batched task, and explain the duration.

- What do we do about output_pk for batched tasks, etc?


Recreating tasks from the database
----------------------------------


Unexpected behaviour
--------------------

-> If you use `task.run()` then the database will not be propagated with information about the task parameters. This is because the task parameters are populated when an event is triggered that the event has started. That event does not get triggered by `task.run()`. Instead, you should use `astra.build([task])` to run the task, which will also build up the dependency graph and make sure all requirements are fulfilled. When the task starts running, the task parameters will be populated to the database.


-> The output interface.