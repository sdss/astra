
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

Let's define the schema for our table, which we will call `astra.random_number_generator` ::

  set search_path to astra;
  drop table if exists astra.random_number_generator;

  create table astra.random_number_generator (
    output_pk int primary key,
    samples real[],
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
  );

The requirements on this table are that it should have an `output_pk`, which is a foreign key referencing the `pk` column in the `astra.output_interface` table. There reason we do this is so that output from an analysis code has a reference key that is unique across all possible result tables.

Load the schema into the database.

Now we can write a Python class that will let us make ORM queries against the database table. You will need to add the following code to the `python/astra/database/astradb.py` file ::

    # The Base and OutputMixin classes are
    # defined in python/astra/database/astradb.py  

    class RandomNumberGenerator(Base, OutputMixin):
        __tablename_ = "random_number_generator"  


Now we can create some tasks. All of the contributed analysis methods to Astra live in the `python/astra/contrib/` folder, or in the `astra.contrib` Python namespace. Normally these contributed analysis packages have a lot of files, and the tasks will live in their own `tasks` sub-folder (e.g., `python/astra/contrib/rng/tasks`), but here we will just make a folder called `python/astra/contrib/rng/` and put the following code in a `__init__.py` file ::

    import astra
    import numpy as np
    from astra.database import astradb
    from astra.tasks import BaseTask
    from astra.tasks.targets import DatabaseTarget
    from time import sleep

    class RandomNumberGeneratorTask(BaseTask):

        """ A task to generate random numbers. """
        
        task_namespace = "RNG"

        seed = astra.IntParameter(description="The random seed to use.")
        draws = astra.IntParameter(
            description="The number of draws to make",
            default=1
        )
        delay_time = astra.IntParameter(
            description="The number of seconds to wait before drawing random numbers.",
            default=0
        )


        def requires(self):
            """ Other tasks that must be completed before this task can be run. """
            return []


        def run(self):
            """ Execute the task. """

            # Wait a little bit.
            sleep(self.delay_time)

            # Set the seed.
            np.random.seed(self.seed)

            # Draw some samples and write them to the database.
            self.output()["database"].write({
                "samples": np.random.normal(size=self.draws)
            })


        def output(self):
            """ The output produced by this task. """
            return {
                "database": DatabaseTarget(astradb.RandomNumberGenerator, self)
            }
            
            

Now we are ready to create and run some tasks. Let's run a simple example ::

    import astra
    from astra.contrib.rng import RandomNumberGeneratorTask

    tasks = [
        RandomNumberGeneratorTask(seed=0),
        RandomNumberGeneratorTask(seed=0, draws=10),
        RandomNumberGeneratorTask(seed=3, draws=2, delay_time=5),
        RandomNumberGeneratorTask(seed=5, draws=1, delay_time=3)
    ]

    astra.build(tasks, local_scheduler=True)


This produces the following output ::

    [<RNG.RandomNumberGeneratorTask(36a542f0)>, <RNG.RandomNumberGeneratorTask(ced8556d)>, <RNG.RandomNumberGeneratorTask(41bbabea)>, <RNG.RandomNumberGeneratorTask(5abe446d)>]
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_36a542f0   has status   PENDING
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_ced8556d   has status   PENDING
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_41bbabea   has status   PENDING
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_5abe446d   has status   PENDING
    INFO: Done scheduling tasks
    INFO: Running Worker with 1 processes
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) running   <RNG.RandomNumberGeneratorTask(36a542f0)>
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) done      <RNG.RandomNumberGeneratorTask(36a542f0)>
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_36a542f0   has status   DONE
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) running   <RNG.RandomNumberGeneratorTask(ced8556d)>
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) done      <RNG.RandomNumberGeneratorTask(ced8556d)>
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_ced8556d   has status   DONE
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) running   <RNG.RandomNumberGeneratorTask(41bbabea)>
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) done      <RNG.RandomNumberGeneratorTask(41bbabea)>
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_41bbabea   has status   DONE
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) running   <RNG.RandomNumberGeneratorTask(5abe446d)>
    INFO: [pid 72852] Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) done      <RNG.RandomNumberGeneratorTask(5abe446d)>
    INFO: Informed scheduler that task   RNG.RandomNumberGeneratorTask_5abe446d   has status   DONE
    INFO: Worker Worker(salt=444353355, workers=1, host=notchpeak21, username=u6020307, pid=72852) was stopped. Shutting down Keep-Alive thread
    INFO: 
    ===== Execution Summary =====

    Scheduled 4 tasks of which:
    * 4 ran successfully:
        - 4 RNG.RandomNumberGeneratorTask(...)

    This progress looks :) because there were no failed tasks or missing dependencies

    ===== Execution Summary =====

You can see that the tasks each have different identifiers (like `36a542f0`, `ced8556d`) that are constructed from the parameters given to that task. Now let's query the database for the results ::

    [u6020307@mwm:astra]$ psql -h operations.sdss.org -d sdss5db -U sdss
    psql (9.6.6, server 12.2)
    WARNING: psql major version 9.6, server major version 12.
            Some psql features might not work.
    Type "help" for help.

    sdss5db=> select * from astra.random_number_generator;
    output_pk |                                                  samples                                                  
    -----------+-----------------------------------------------------------------------------------------------------------
            5 | {1.7640524}
            6 | {1.7640524,0.4001572,0.978738,2.2408931,1.867558,-0.9772779,0.95008844,-0.1513572,-0.10321885,0.41059852}
            7 | {1.7886285,0.43650985}
            8 | {0.4412275}


    sdss5db=> select t.pk, t.task_module, t.task_id, t.duration, rng.samples from astra.task as t, astra.random_number_generator as rng where t.output_pk = rng.output_pk;
    pk |    task_module    |                task_id                 |   duration   |                                                  samples                                                  
    ----+-------------------+----------------------------------------+--------------+-----------------------------------------------------------------------------------------------------------
    1 | astra.contrib.rng | RNG.RandomNumberGeneratorTask_36a542f0 | 0.0099208355 | {1.7640524}
    2 | astra.contrib.rng | RNG.RandomNumberGeneratorTask_ced8556d | 0.0065267086 | {1.7640524,0.4001572,0.978738,2.2408931,1.867558,-0.9772779,0.95008844,-0.1513572,-0.10321885,0.41059852}
    3 | astra.contrib.rng | RNG.RandomNumberGeneratorTask_41bbabea |    5.0129633 | {1.7886285,0.43650985}
    4 | astra.contrib.rng | RNG.RandomNumberGeneratorTask_5abe446d |    3.0119936 | {0.4412275}


Here you can see that the first two tasks took almost no time at all, but the third and fourth tasks took longer because of the `time_delay` parameter we gave ::

    sdss5db=> select t.task_id, t.duration, p.parameter_name, p.parameter_value from astra.parameter as p, astra.task_parameter as tp, astra.task as t where t.pk = 3 and t.pk = tp.task_pk and tp.parameter_pk = p.pk;
                    task_id                 | duration  |   parameter_name    | parameter_value 
    ----------------------------------------+-----------+---------------------+-----------------
    RNG.RandomNumberGeneratorTask_41bbabea | 5.0129633 | astra_version_major | 0
    RNG.RandomNumberGeneratorTask_41bbabea | 5.0129633 | astra_version_minor | 1
    RNG.RandomNumberGeneratorTask_41bbabea | 5.0129633 | seed                | 3
    RNG.RandomNumberGeneratorTask_41bbabea | 5.0129633 | draws               | 2
    RNG.RandomNumberGeneratorTask_41bbabea | 5.0129633 | delay_time          | 5

This shows how we can track the parameters given to every task, without having to write any additional code. All we need to do is to make a cross-match between the `task`, `parameter`, and `task_parameter` tables. And while we didn't specify `astra_version_major` and `astra_version_minor` as parameters to our `RandomNumberGeneratorTask` task class, these parameters are inherited for every Astra task so we can track any changes in results with time.





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