import unittest
from time import sleep


class TestTaskTiming(unittest.TestCase):

    def test_task_timing(self):

        pre_execute_sleep_length = 3
        post_execute_sleep_length = 2

        from astra.base import ExecutableTask, Parameter
        from astra.database.astradb import database, Task, Bundle, TaskBundle

        # Create tables if they don't exist.
        if not database.table_exists(Task):
            with database.atomic():
                models = (Task, Bundle, TaskBundle)
                database.create_tables(models) 

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

        ids = [task.id for task in bundled_task.context["tasks"]]
        tasks = Task.select().where(Task.id.in_(ids))

        # Check timings.    
        places = 1
        for task, sl in zip(tasks, sleep_length):
            self.assertAlmostEqual(task.time_pre_execute_bundle, pre_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_pre_execute, pre_execute_sleep_length / N, places=places)
            self.assertAlmostEqual(task.time_execute_task, sl, places=places)
            self.assertEqual(task.time_execute, task.time_execute_bundle / N + task.time_execute_task)
            self.assertAlmostEqual(task.time_post_execute_bundle, post_execute_sleep_length, places=places)
            self.assertAlmostEqual(task.time_post_execute, post_execute_sleep_length / N, places=places)

            self.assertEqual(task.time_total, task.time_pre_execute + task.time_execute + task.time_post_execute)
            
        return None
