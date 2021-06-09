
.. title:: Database targets

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Database targets

:tocdepth: 1

.. rubric:: :header_no_toc:`Database targets`

Test doc update

In Astra you can write the output of a task to a file, write the result to a database, or both.
This page provides a primer or reference for how to define a database target row.

Let's pretend we have some task that we want to write the output to a database.
The way we do this in Astra is to define an *output* of a task to be a :py:mod:`astra.tasks.targets.DatabaseTarget` object.

Here's what it might look like::

    import astra
    from astra.tasks.base import BaseTask
    from astra.tasks.targets import DatabaseTarget
    from sqlalchemy import Column, Float

    # Let's first define a database target.
    class MyTaskResultTarget(DatabaseTarget):
        
        # These are the expected outputs from the task.
        foo = Column("foo", Float)
        result = Column("result", Float)


    # Now define the task.
    class MyTask(BaseTask):

        # Define some parameters for the task.
        a = astra.FloatParameter()
        b = astra.IntParameter()
        c = astra.FloatParameter()

        def run(self):
            # Calculate or do something.
            result = self.a + self.b * self.c
            foo = max([self.a, self.b, self.c])
            
            # Write some outputs.
            self.output().write(dict(result=result, foo=foo))


        def output(self):
            return MyTaskResultTarget(self)



Now if we wanted to run that task::

    task = MyTask(a=3.5, b=4, c=6)

    task.run()

And access the results afterwards::

    >>> # Has this task been executed?
    >>> print(task.exists())
    True
    >>> # Let's look at the results.
    >>> print(task.output().read())
    ('MyTask_0c64e8f4c6', 0, 1, 3.5, 4, 6.0, 6.0, 27.5)
    >>> # Let's look as if it was a dictionary:
    >>> for key, value in task.output().read(as_dict=True).items():
    >>>     print(f"{key}: {value}")
    task_id: MyTask_0c64e8f4c6
    astra_version_major: 0
    astra_version_minor: 1
    a: 3.5
    b: 4
    c: 6.0
    foo: 6.0
    result: 27.5
