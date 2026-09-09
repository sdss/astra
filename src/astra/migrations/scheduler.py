"""
Migration task scheduler with DAG-based dependency resolution.

This module provides a scheduler for running migration tasks with:
- Dependency tracking between tasks
- Resource conflict avoidance (prevents concurrent writes to same table)
- Hierarchical progress reporting with subtasks
"""

import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable, Set, Dict, Any, Optional

from rich.progress import Progress

from astra.migrations.utils import ProgressContext
from astra.utils import log


def _drain_and_close(queue) -> None:
    """Drain any remaining items from a multiprocessing queue and close it.

    Migration tasks (and ``_task_wrapper``) can both emit a completion
    sentinel, so a queue may still hold messages once the scheduler has seen
    the first one. Draining and closing here prevents the queue's feeder
    thread from lingering and blocking interpreter shutdown.
    """
    try:
        while True:
            queue.get(block=False)
    except mp.queues.Empty:
        pass
    except Exception:
        pass
    try:
        queue.close()
        queue.cancel_join_thread()
    except Exception:
        pass


def _task_wrapper(target, *args, **kwargs):
    """Wrapper to run a migration task in a subprocess and handle exceptions."""
    # Reconnect database in subprocess to avoid stale connections
    from astra.models.base import database
    if not database.is_closed():
        database.close()
    database.connect()

    try:
        target(*args, **kwargs)
    except Exception as e:
        q = kwargs.get("queue", None)
        e.add_note(f"\n\nRaised in {target.__name__}()")
        if q is not None:
            # Get the underlying queue from ProgressContext
            if hasattr(q, '_queue') and q._queue is not None:
                q._queue.put(e)
            elif hasattr(q, 'put'):
                q.put(e)
    else:
        q = kwargs.get("queue", None)
        if q is not None:
            # Get the underlying queue from ProgressContext
            if hasattr(q, '_queue') and q._queue is not None:
                q._queue.put(Ellipsis)
            elif hasattr(q, 'put'):
                q.put(Ellipsis)


@dataclass
class MigrationTask:
    """
    A migration task with dependencies and resource constraints.

    Attributes:
        name: Unique identifier for this task
        func: The function to execute
        description: Human-readable description for progress display
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        depends_on: Set of task names that must complete before this task runs
        writes_to: Set of table names this task writes to (prevents concurrent writes)
        exclusive: If True, no other tasks run while this one runs (for tasks with internal pools)
    """
    name: str
    func: Callable
    description: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    depends_on: Set[str] = field(default_factory=set)
    writes_to: Set[str] = field(default_factory=set)
    exclusive: bool = False


class MigrationScheduler:
    """
    DAG-based task scheduler that respects dependencies and resource constraints.

    Scheduling rules:
    1. A task starts only when all its dependencies have completed
    2. Tasks writing to the same table don't run concurrently
    3. Exclusive tasks run alone (no other tasks concurrent)

    Progress reporting:
    - Each task gets a progress bar in the CLI
    - Tasks can create subtasks for multi-phase operations
    - Subtasks appear indented below their parent task
    """

    def __init__(self, tasks: Dict[str, MigrationTask], progress: Progress, linger_time: float = 10.0):
        self.tasks = tasks
        self.progress = progress
        self.linger_time = linger_time

        # State tracking
        self.pending: Set[str] = set(tasks.keys())
        self.running: Dict[str, tuple] = {}  # name -> (process, progress_task, queue)
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.recently_completed: Dict[int, float] = {}  # progress_task_id -> completion_time
        self.subtasks: Dict[str, int] = {}  # subtask_id -> progress_task_id

        # Resource tracking
        self.tables_being_written: Set[str] = set()
        self.exclusive_running: bool = False

    def get_ready_tasks(self) -> list:
        """Return tasks that are ready to start based on dependencies and resources."""
        ready = []

        # If an exclusive task is running, nothing else can start
        if self.exclusive_running:
            return []

        for name in list(self.pending):
            task = self.tasks[name]

            # Check dependencies
            if not task.depends_on.issubset(self.completed):
                continue

            # Check for write conflicts
            if task.writes_to & self.tables_being_written:
                continue

            # If this task is exclusive, only start it if nothing else is running
            if task.exclusive and self.running:
                continue

            ready.append(name)

            # If this task is exclusive, don't add any more
            if task.exclusive:
                break

        return ready

    def start_task(self, name: str) -> None:
        """Start a task in a subprocess."""
        task = self.tasks[name]
        queue = mp.Queue()

        # Wrap the queue with ProgressContext for hierarchical progress support
        progress_ctx = ProgressContext(queue, task_id=name)

        kwargs = dict(task.kwargs)
        kwargs["queue"] = progress_ctx

        process = mp.Process(
            target=_task_wrapper,
            args=(task.func,) + task.args,
            kwargs=kwargs
        )
        process.start()

        progress_task = self.progress.add_task(description=task.description, total=None)
        self.running[name] = (process, progress_task, queue)
        self.pending.remove(name)

        # Track resources
        self.tables_being_written |= task.writes_to
        if task.exclusive:
            self.exclusive_running = True

    def check_running_tasks(self) -> None:
        """Check status of running tasks and handle completions."""
        for name in list(self.running.keys()):
            process, progress_task, queue = self.running[name]
            task = self.tasks[name]

            try:
                msg = queue.get(block=False)
                if msg is Ellipsis:
                    # Task completed successfully - mark complete but keep visible
                    self.progress.update(progress_task, description=f"[green]✓[/green] {task.description}")

                    # Join with a timeout so a child that fails to shut down
                    # cleanly (e.g. a leaked process pool deadlocking at exit)
                    # can't hang the whole scheduler. Fall back to terminate.
                    process.join(timeout=30)
                    if process.is_alive():
                        log.warning(
                            f"Task {name!r} reported completion but did not exit "
                            f"within 30s; terminating."
                        )
                        process.terminate()
                        process.join(timeout=5)

                    # Drain and close the queue so leftover messages (e.g. the
                    # duplicate completion sentinel) don't leave an orphaned
                    # feeder thread alive at interpreter exit.
                    _drain_and_close(queue)

                    # Clean up any subtask state for this task
                    for subtask_id in list(self.subtasks.keys()):
                        if subtask_id.startswith(name + "."):
                            del self.subtasks[subtask_id]

                    del self.running[name]
                    self.completed.add(name)
                    self.recently_completed[progress_task] = time.time()

                    # Release resources
                    self.tables_being_written -= task.writes_to
                    if task.exclusive:
                        self.exclusive_running = False

                elif isinstance(msg, Exception):
                    # Task failed - terminate other running processes first
                    log.exception(msg)
                    for other_name, (other_proc, other_progress_task, _) in list(self.running.items()):
                        if other_name != name:
                            other_proc.terminate()
                            other_proc.join(timeout=5)
                            self.progress.update(other_progress_task, visible=False)

                    del self.running[name]
                    self.failed.add(name)

                    # Release resources
                    self.tables_being_written -= task.writes_to
                    if task.exclusive:
                        self.exclusive_running = False

                    raise msg

                elif isinstance(msg, tuple):
                    # New hierarchical progress protocol
                    # Subtasks are shown inline in the parent task's description
                    cmd, *args = msg
                    if cmd == "add_subtask":
                        subtask_id, parent_id, kwargs = args
                        subtask_desc = kwargs.get('description', '')
                        subtask_total = kwargs.get('total')
                        # Store subtask info - we'll update the parent task
                        self.subtasks[subtask_id] = {
                            'description': subtask_desc,
                            'total': subtask_total,
                            'completed': 0,
                            'parent_task': progress_task,
                            'base_description': task.description
                        }
                        # Update parent task to show subtask
                        self.progress.update(
                            progress_task,
                            description=f"{task.description}: {subtask_desc}",
                            total=subtask_total,
                            completed=0
                        )
                        if subtask_total is not None:
                            self.progress.reset(progress_task)

                    elif cmd == "update":
                        task_id, kwargs = args
                        if task_id in self.subtasks:
                            # Update subtask info and reflect on parent
                            subtask_info = self.subtasks[task_id]
                            if 'description' in kwargs:
                                subtask_info['description'] = kwargs['description']
                            if 'total' in kwargs:
                                subtask_info['total'] = kwargs['total']
                            if 'completed' in kwargs:
                                subtask_info['completed'] = kwargs['completed']
                            if 'advance' in kwargs:
                                subtask_info['completed'] = subtask_info.get('completed', 0) + kwargs['advance']

                            # Update parent task display
                            update_kwargs = {}
                            if 'description' in kwargs:
                                update_kwargs['description'] = f"{subtask_info['base_description']}: {kwargs['description']}"
                            if 'total' in kwargs:
                                update_kwargs['total'] = kwargs['total']
                            if 'completed' in kwargs:
                                update_kwargs['completed'] = kwargs['completed']
                            if 'advance' in kwargs:
                                update_kwargs['advance'] = kwargs['advance']

                            if update_kwargs:
                                self.progress.update(subtask_info['parent_task'], **update_kwargs)
                        elif task_id == name:
                            self.progress.update(progress_task, **kwargs)
                            if kwargs.get("completed") == 0:
                                self.progress.reset(progress_task)

                    elif cmd == "complete_subtask":
                        subtask_id = args[0]
                        if subtask_id in self.subtasks:
                            subtask_info = self.subtasks[subtask_id]
                            # Restore parent task description (subtask complete)
                            self.progress.update(
                                subtask_info['parent_task'],
                                description=subtask_info['base_description']
                            )
                            del self.subtasks[subtask_id]
                else:
                    # Legacy dict-style progress update (backward compat)
                    self.progress.update(progress_task, **msg)
                    if msg.get("completed") == 0:
                        self.progress.reset(progress_task)

            except mp.queues.Empty:
                pass

    def hide_old_completed_tasks(self) -> None:
        """Hide completed tasks that have been visible for longer than linger_time."""
        now = time.time()
        to_remove = []
        for progress_task, completed_at in self.recently_completed.items():
            if now - completed_at >= self.linger_time:
                self.progress.update(progress_task, visible=False)
                to_remove.append(progress_task)
        for task_id in to_remove:
            del self.recently_completed[task_id]

    def run(self) -> None:
        """Execute all tasks respecting dependencies and constraints."""
        while self.pending or self.running or self.recently_completed:
            # Start any tasks that are ready
            for name in self.get_ready_tasks():
                self.start_task(name)

            # Check on running tasks
            self.check_running_tasks()

            # Hide completed tasks after linger time
            self.hide_old_completed_tasks()

            # Small sleep to avoid busy-waiting
            time.sleep(0.05)


def get_satisfiable_tasks(tasks: Dict[str, MigrationTask]) -> Dict[str, MigrationTask]:
    """Remove tasks whose dependencies can never be satisfied."""
    available = set(tasks.keys())
    changed = True
    while changed:
        changed = False
        for name, task in list(tasks.items()):
            if not task.depends_on.issubset(available):
                del tasks[name]
                available.remove(name)
                changed = True
    return tasks
