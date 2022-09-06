import re
from getpass import getuser
from subprocess import call, Popen, PIPE


def get_slurm_queue():
    """Get a list of jobs currently in the Slurm queue."""

    pattern = (
        "(?P<job_id>\d+)+\s+(?P<name>[-\w\d_\.]+)\s+(?P<user>[\w\d]+)\s+(?P<group>\w+)"
        "\s+(?P<account>[-\w]+)\s+(?P<partition>[-\w]+)\s+(?P<time_limit>[-\d\:]+)\s+"
        "(?P<time_left>[-\d\:]+)\s+(?P<status>\w*)\s+(?P<nodelist>[\w\d\(\)]+)"
    )
    process = Popen(
        [
            "/uufs/notchpeak.peaks/sys/installdir/slurm/std/bin/squeue",
            "--account=sdss-np,notchpeak-gpu,sdss-np-fast",
            '--format="%14i %50j %10u %10g %13a %13P %11l %11L %2t %R"',
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()

    # Parse the output.
    return [match.groupdict() for match in re.finditer(pattern, output)]


def get_slurm_job(name, user=None):
    """
    Get the status of a Slurm job given its name.

    :param name:
        The name (or label) given to the Slurm job.

    :param user: [optional]
        The user who submitted the job. If `None` is given it will default to the current user.
    """
    user = user or getuser()
    jobs = get_slurm_queue()

    for job in jobs:
        if job["name"] == name and job["user"] == user:
            return job
    else:
        raise KeyError(f"No slurm job '{name}' by '{user}' found among {jobs}")


def cancel_slurm_job_given_name(name, user=None):
    """
    Cancel a Slurm job matching the given name and user.

    :param name:
        The name (or label) given to the Slurm job.

    :param user: [optional]
        The user who submitted the job. If `None` is given it will default to the current user.
    """

    job = get_slurm_job(name, user)

    call(["scancel", job["job_id"]])

    return None
