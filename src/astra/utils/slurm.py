
import re
import os
from getpass import getuser
from subprocess import check_output, call, Popen, PIPE

from astra.utils import expand_path



def get_queue():
    """Get a list of jobs currently in the Slurm queue."""

    pattern = (
        r"(?P<job_id>\d+)+\s+(?P<name>[-\w\d_\.]+)\s+(?P<user>[\w\d]+)\s+(?P<group>\w+)"
        r"\s+(?P<account>[-\w]+)\s+(?P<partition>[-\w]+)\s+(?P<time_limit>[-\d\:]+)\s+"
        r"(?P<time_left>[-\d\:]+)\s+(?P<status>\w*)\s+(?P<nodelist>[\w\d\(\)]+)"
    )
    process = Popen(
        [
            "/uufs/notchpeak.peaks/sys/installdir/slurm/std/bin/squeue",
            "--account=sdss-np,sdss-kp,notchpeak-gpu,sdss-np-fast",
            '--format="%14i %50j %10u %10g %13a %13P %11l %11L %2t %R"',
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()

    # Parse the output.
    queue = dict()
    for job in [match.groupdict() for match in re.finditer(pattern, output)]:
        queue[int(job.get("job_id"))] = job
    return queue






class SlurmTask:

    def __init__(self, commands):
        self.commands = commands
        return None
    
    def set_meta(self, directory, node_index, task_index):
        self.directory = directory
        self.task_index = task_index
        self.node_index = node_index or 1
        return None

    def write(self):
        path = expand_path(f"{self.directory}/node{self.node_index:0>2.0f}_task{self.task_index:0>2.0f}.slurm")
        with open(path, "w") as fp:
            fp.write("\n".join(self.commands))
        return path
    

class SlurmJob:

    def __init__(self, tasks, job_name, account, partition=None, walltime="24:00:00", mem=None, ppn=None, gres=None, ntasks=None, nodes=1, node_index=None, dir=None):
        self.account = account
        self.partition = partition or account
        self.walltime = walltime
        self.job_name = job_name
        self.tasks = tasks
        self.nodes = nodes
        self.gres = gres
        self.mem = mem
        self.ppn = ppn
        self.dir = dir or expand_path(f"$PBS/{self.job_name}")
        if ntasks is None and account is not None:
            ntasks = {
                "sdss-kp": 16,
                "sdss-np": 64
            }.get(account.lower(), 16)
                
        self.ntasks = ntasks
        self.node_index = node_index
        for j, task in enumerate(self.tasks, start=1):
            task.set_meta(self.dir, self.node_index or 1, j)
        os.makedirs(self.dir, exist_ok=True)
        return None


    def write(self):
        if self.node_index is None:
            node_index = 1
            node_index_suffix = ""
        else:
            node_index = self.node_index
            node_index_suffix = f"_{self.node_index:0>2.0f}"

        contents = [
            "#!/bin/bash",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.ntasks}",
        ]
        if self.ppn is not None:
            contents.append(f"#SBATCH --ppn={self.ppn}")
        if self.mem is not None:
            contents.append(f"#SBATCH --mem={self.mem}")
        if self.gres is not None:
            contents.append(f"#SBATCH --gres={self.gres}")
        
        contents.extend([
            f"#SBATCH --time={self.walltime}",
            f"#SBATCH --job-name={self.job_name}{node_index_suffix}",
            f"#SBATCH --output={self.dir}/slurm_%A.out",
            f"#SBATCH --err={self.dir}/slurm_%A.err",
            f"# ------------------------------------------------------------------------------",
            "export CLUSTER=1"
        ])
        for task in self.tasks:
            #log_prefix = f"{self.dir}/node{node_index:0>2.0f}_task{task_index:0>2.0f}"
            contents.append(f"source {task.write()} &")
        contents.extend(["wait", "echo \"Done\""])

        node_path = expand_path(f"{self.dir}/node{node_index:0>2.0f}.slurm")
        with open(node_path, "w") as fp:
            fp.write("\n".join(contents))

        return node_path


    def submit(self):
        slurm_path = self.write()
        output = check_output(["sbatch", slurm_path]).decode("ascii")
        job_id = int(output.split()[3])
        return job_id
