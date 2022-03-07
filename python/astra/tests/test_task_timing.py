import os
import unittest
from time import sleep
import numpy as np

class TestTaskTiming(unittest.TestCase):


    def setUp(self):
        self.env_var_name = "ASTRA_DATABASE_URL"

        self.has_original = self.env_var_name in os.environ
        self.original = os.environ.get(self.env_var_name, None)

        os.environ["ASTRA_DATABASE_URL"] = "sqlite:///:memory:"

    def test_task_timing(self):

        pre_execute_sleep_length = 3
        post_execute_sleep_length = 2

        from astra.base import ExecutableTask, Parameter
        from astra.database.astradb import database, Task, Bundle, TaskBundle

        models = (Task, Bundle, TaskBundle)
        database.create_tables(models) # since we are using an in-memory database

        class TestTask(ExecutableTask):

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

        # Check timings.    
        places = 1
        for task, sl in zip(Task.select(), sleep_length):
            self.assertAlmostEqual(task.time_pre_execute_bundle, pre_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_pre_execute, pre_execute_sleep_length / N, places=places)
            self.assertAlmostEqual(task.time_execute_task, sl, places=places)
            self.assertEqual(task.time_execute, task.time_execute_bundle / N + task.time_execute_task)
            self.assertAlmostEqual(task.time_post_execute_bundle, post_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_post_execute, post_execute_sleep_length / N, places=places)

            self.assertEqual(task.time_total, task.time_pre_execute + task.time_execute + task.time_post_execute)
            
        return None
    
    def tearDown(self):
        if self.has_original:
            os.environ[self.env_var_name] = self.original
        else:
            del os.environ[self.env_var_name]

