
.. title:: Database

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Database

:tocdepth: 1

.. rubric:: :header_no_toc:`Database`


The primary Astra database is hosted with the existing SDSS-V PostgreSQL database server that is 
hosted at Utah. We use the general utilities and functionality for connecting to SDSS databases 
from `sdssdb` , which uses object-relational mapping (ORM)

If you want to run Astra locally or direct the database connection to somewhere else, you can do so
by changing the settings in your `astra` profile in your `sdssdb` configuration file.
The database that Astra connects to will need to have the Astra database schema, which is stored in
`schema/astradb/astradb.sql` and can be loaded into PostgreSQL with the terminal command ::

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


The `task` table
-----------------

+------------------------+------------+------------------------------------------------------------+
| Column name            | Type       | Description                                                |
+========================+============+============================================================+
| `pk`                   | serial primary key | Test |
+------------------------+------------+------------------------------------------------------------+
| `task_module`          | text | test |
+------------------------+------------+------------------------------------------------------------+

