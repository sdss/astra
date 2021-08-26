
from astra.operators.base import AstraOperator
from astra.operators.apogee_visit import ApVisitOperator
from collections import OrderedDict

def test_bash_command_kwargs_for_astraoperator():

    bash_command = "sleep"
    bash_command_kwargs = OrderedDict([
        ("this", "that"),
        ("moo", 3)
    ])

    A = AstraOperator(task_id="A")
    B = AstraOperator(
        task_id="B",
        bash_command=bash_command,
        bash_command_kwargs=bash_command_kwargs
    )

    C = ApVisitOperator(
        task_id="C",
        bash_command=bash_command,
        bash_command_kwargs=bash_command_kwargs
    )
    
    D = ApVisitOperator(
        task_id="D",
        release="sdss5",
        bash_command=bash_command,
        bash_command_kwargs=bash_command_kwargs
    )

    assert A.bash_command_line_arguments == ""
    assert B.bash_command_line_arguments == "--this that --moo 3"
    assert C.common_task_parameters()["bash_command_kwargs"] == bash_command_kwargs

    assert "release" not in C.common_task_parameters()
    assert D.common_task_parameters()["release"] == "sdss5"