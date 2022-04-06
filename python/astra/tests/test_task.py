import unittest
from astra.base import ExecutableTask, Parameter, TupleParameter
from time import sleep



class TestTaskBehaviour(unittest.TestCase):

    def test_parameter_inheritance(self):
        class Parent(ExecutableTask):
            A = Parameter()

        class Child(Parent):
            B = Parameter()

        with self.assertRaises(TypeError):
            Child(B=1) # should be missing A param
        
        Child(B=1, A=0) # should work

        class Parent(ExecutableTask):
            A = Parameter(default=0)
        
        class Child(Parent):
            B = Parameter(default=5)

        task = Child()
        self.assertEqual(task.A, 0)
        self.assertEqual(task.B, 5)


    def test_pre_post_execution_counts(self):

        global NUM_PRE_EXECUTIONS 
        global NUM_POST_EXECUTIONS

        NUM_PRE_EXECUTIONS = 0
        NUM_POST_EXECUTIONS = 0

        class Parent(ExecutableTask):
            A = Parameter(default=0)

            def post_execute(self):
                global NUM_POST_EXECUTIONS
                NUM_POST_EXECUTIONS += 1
                print("in post_execute of parent")

        class Child(Parent):
            B = Parameter()
            
            def pre_execute(self):
                global NUM_PRE_EXECUTIONS
                NUM_PRE_EXECUTIONS += 1
                print("in pre_execute of child")

            def execute(self):
                for item in self.iterable():
                    sleep(1)

        class StubbornChild(Parent):
            B = Parameter()

            def post_execute(self):
                print("in post execute of child")
                self.child_won = True

        # this has a pre_execute from child, and post_execute from parent
        # each should only be called once
        C = Child(B=1)
        C.execute()

        self.assertEqual(NUM_PRE_EXECUTIONS, 1)
        self.assertEqual(NUM_POST_EXECUTIONS, 1)

        # If a child class has the pre/post execute, then the parent one will
        # not be executed (unless explicitly called)
        D = StubbornChild(B=1)
        D.execute()
        self.assertTrue(D.child_won)
        self.assertEqual(NUM_POST_EXECUTIONS, 1)


    def test_skip_pre_execute(self):

        class DummyTask(ExecutableTask):
            def pre_execute(self):
                return 5
            def execute(self):
                return 4

        A = DummyTask()
        A.execute(decorate_pre_execute=False)

        # without the decorator, we have no idea how much time it took
        self.assertNotIn("time_pre_execute", A.context["timing"])

        B = DummyTask()
        B.execute(decorate_pre_execute=False, decorate_post_execute=False)
        self.assertNotIn("time_pre_execute", B.context["timing"])
        self.assertNotIn("time_post_execute", B.context["timing"])

        # This should be true regardless whether we defined pre/post_execute or not
        class DummyTask(ExecutableTask):
            def execute(self):
                return 4        
        
        C = DummyTask()
        C.execute(decorate_pre_execute=False, decorate_post_execute=False)
        self.assertNotIn("time_pre_execute", C.context["timing"])
        self.assertNotIn("time_post_execute", C.context["timing"])

        D = DummyTask()
        D.execute(decorate_execute=False)
        self.assertNotIn("timing", D.context)


    def test_task_bundling(self):

        class DummyTask(ExecutableTask):
            A = Parameter()
            B = Parameter(bundled=True)
            C = Parameter(default=None)

        t = DummyTask(A=1, B="hello", C=3) # ok
        self.assertEqual(t.bundle_size, 1)

        t = DummyTask(A=(1, 2), B="hello", C=(3, 3)) # ok, all non-bundled are same length
        self.assertEqual(t.bundle_size, 2)
        t = DummyTask(A=(1, 2, 3), B="hello") # ok, 3 tasks, all non-bundled are same length, C gets default
        self.assertEqual(t.bundle_size, 3)

        # check parameters as expected:
        for i, (sub_task, dp, p) in enumerate(t.iterable()):
            self.assertEqual(sub_task.parameters["A"], i + 1)
            self.assertEqual(p["A"], i + 1)
            self.assertIsNone(sub_task.parameters["C"])
            self.assertEqual(sub_task.parameters["B"], "hello")

        with self.assertRaises(TypeError):
            DummyTask(A=(1,2,3), B=("this", "that"), C=(1,2,3)) # not ok, a bundled param is different length.

        # not ok, given params are different lengths
        with self.assertRaises(ValueError):
            DummyTask(A=(1, 2), B="moo", C=(1,2,3))

        class DummyTask2(DummyTask):
            D = TupleParameter(default=(0, 1))
            E = TupleParameter(bundled=True)

        # tuple and dict parameters are taken AS-IS. their lengths are not considered
        t = DummyTask2(A=1, B=34, E=10)
        t = DummyTask2(A=(1, 2, 3), B=34, C=(3, 4, 5), E=10)
        self.assertEqual(t.bundle_size, 3)
        self.assertEqual(t.D, (0, 1))
        self.assertEqual(t.E, 10)

'''        



import unittest
from astra.base import ExecutableTask, Parameter, TupleParameter
from time import sleep

class DummyTask(ExecutableTask):
    A = Parameter()
    B = Parameter(bundled=True)
    C = Parameter(default=None)


DummyTask(A=1, B="hello", C=3) # ok
DummyTask(A=(1, 2), B="hello", C=(3, 3)) # ok, all non-bundled are same length
foo = DummyTask(A=(1,2,3), B="hello") # ok, 3 tasks, all non-bundled are same length, C gets default

# Not OK, the lengths of A and C are different.
#with self.assertRaises(ValueError):
#    DummyTask(A=(1, 2), B="moo", C=(1,2,3))

class DummyTask2(DummyTask):
    D = TupleParameter(default=(0, 1))
    E = TupleParameter(bundled=True)
# tuple and dict parameters are taken AS-IS. their lengths are not considered



# not ok, a bundled param is different length.

#with self.assertRaises(TypeError):
#    DummyTask(A=(1,2,3), B=("this", "that"), C=(1,2,3)) 

'''