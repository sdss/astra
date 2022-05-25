import os
import unittest
import json
from astra.base import TaskInstance, Parameter, TupleParameter
from astra.database.astradb import database, create_tables, Task, Bundle, TaskBundle, DataProduct, TaskInputDataProducts
from time import sleep
from tempfile import mkstemp








class TestTaskBehaviour(unittest.TestCase):

    def setUp(self):
        # Create tables if they don't exist.
        if not database.table_exists(Task):
            create_tables()

    def test_parameter_inheritance(self):
        class Parent(TaskInstance):
            A = Parameter()

        class Child(Parent):
            B = Parameter()

        with self.assertRaises(TypeError):
            Child(B=1) # should be missing A param
        
        Child(B=1, A=0) # should work

        class Parent(TaskInstance):
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

        class Parent(TaskInstance):
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

        class DummyTask(TaskInstance):
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
        class DummyTask(TaskInstance):
            def execute(self):
                return 4        
        
        C = DummyTask()
        C.execute(decorate_pre_execute=False, decorate_post_execute=False)
        self.assertNotIn("time_pre_execute", C.context["timing"])
        self.assertNotIn("time_post_execute", C.context["timing"])

        D = DummyTask()
        D.execute(decorate=False)
        self.assertNotIn("timing", D.context)


    def test_task_bundling(self):

        class DummyTask(TaskInstance):
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


    def test_task_status_handling(self):

        class DummyTask(TaskInstance):
            A = Parameter()

            def execute(local_self):
                for i, (item, *_) in enumerate(local_self.iterable()):
                    self.assertEqual(item.status.description, "running")
                    sleep(i)
            

        t = DummyTask(A=(1, 2, ))
        
        t.get_or_create_context()
        self.assertEqual(t.context["bundle"].status.description, "created")
        for item in t.context["tasks"]:
            self.assertEqual(item.status.description, "created")

        t.execute()

        self.assertEqual(t.context["bundle"].status.description, "completed")
        for item in t.context["tasks"]:
            self.assertEqual(item.status.description, "completed")
            self.assertIsNotNone(item.completed)
        
        for item, *_ in t.iterable():
            self.assertEqual(item.status.description, "completed")
            self.assertIsNotNone(item.completed)

    def test_task_created_and_completed(self):

        class DummyTask(TaskInstance):
            A = Parameter()
            B = Parameter(bundled=True)
            C = Parameter(default=None)

        t = DummyTask(A=(1, 2, 3), B=34, C=(3, 4, 5))
        t.execute()

        for task in t.context["tasks"]:
            self.assertEqual(task.name, "astra.tests.test_task.DummyTask")
            self.assertIsNotNone(task.created)
            self.assertIsNotNone(task.completed)


    def test_task_timing(self):

        pre_execute_sleep_length = 3
        post_execute_sleep_length = 2

        class TestTask(TaskInstance):

            sleep_length = Parameter("sleep_length", default=0)

            def execute(self):
                for task, input_data_products, parameters in self.iterable():
                    print(f"In {self} with {task} and sleep={parameters['sleep_length']}")
                    sleep(parameters["sleep_length"])

            def pre_execute(self):
                print(f"in pre-execute")
                sleep(pre_execute_sleep_length)
            
            def post_execute(self):
                print(f"in post execute")
                sleep(post_execute_sleep_length)


        sleep_length = [0, 1, 5, 6]
        N = len(sleep_length)
        bundled_task = TestTask(sleep_length=sleep_length)
        bundled_task.execute()

        ids = [task.id for task in bundled_task.context["tasks"]]
        tasks = Task.select().where(Task.id.in_(ids))

        # Check timings.    
        places = 1
        for task, sl in zip(tasks, sleep_length):
            self.assertAlmostEqual(task.time_pre_execute_bundle_overhead, pre_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_pre_execute, pre_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_pre_execute_task, 0, places=places)
            
            self.assertAlmostEqual(task.time_execute_task, sl, places=places)
            self.assertAlmostEqual(task.time_execute, sum(sleep_length), places=places)
            self.assertAlmostEqual(task.time_execute_bundle_overhead, 0, places=places)
            
            self.assertAlmostEqual(task.time_post_execute_bundle_overhead, post_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_post_execute, post_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_post_execute_task, 0, places=places)

            self.assertEqual(task.time_total, task.time_pre_execute + task.time_execute + task.time_post_execute)
            
        return None


    def test_data_product_parsing(self):

        class DummyTask(TaskInstance):
            pass
        
        # create some temporary paths
        temp_paths = [mkstemp()[1] for i in range(5)]
        for temp_path in temp_paths:
            self.assertIsInstance(temp_path, str)
            self.assertTrue(os.path.exists(temp_path))

        # Check that these have been created as data products.
        def checker(task):
            for (task, data_products, parameters) in task.iterable():
                for data_product, temp_path in zip(data_products, temp_paths):
                    self.assertIsInstance(data_product, DataProduct)
                    self.assertEqual(data_product.path, temp_path)
                    self.assertEqual(data_product.filetype, "full")

                    self.assertEqual(data_product.kwargs, dict(full=temp_path))

        # Check with paths
        A = DummyTask(input_data_products=temp_paths)
        A.execute()
        checker(A)

        data_products = A.context["input_data_products"]

        # Try with DataProduct
        B = DummyTask(input_data_products=data_products)
        B.execute()
        checker(B)

        # Try with ids
        ids = [dp.id for dp in data_products]
        C = DummyTask(input_data_products=ids)
        C.execute()
        checker(C)

        # Try with json'ified IDS
        D = DummyTask(input_data_products=json.dumps(ids))
        D.execute()
        checker(D)

        # Check with single path
        E = DummyTask(input_data_products=temp_paths[0])
        E.execute()
        checker(E)

        F = DummyTask(input_data_products=ids[0])
        F.execute()
        checker(F)

        G = DummyTask(input_data_products=json.dumps(ids[0]))
        G.execute()
        checker(G)

        H = DummyTask(input_data_products=data_products[0])
        H.execute()
        checker(H)
