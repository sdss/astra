import os
import subprocess
from astra.utils import log, expand_path


def execute(pwd, timeout=None):
        
    pwd = expand_path(pwd)
    stdout_path = os.path.join(pwd, "stdout")
    stderr_path = os.path.join(pwd, "stdout")
    
    log.info(f"Starting ferre in {pwd} with timeout of {timeout}")
    try:
        with open(stdout_path, "w") as stdout:
            with open(stderr_path, "w") as stderr:
                process = subprocess.run(
                    ["ferre.x"],
                    cwd=pwd,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout, 
                )
    except subprocess.TimeoutExpired:
        log.exception(f"FERRE has timed out in {pwd}")
    except:
        log.exception(f"Exception when calling FERRE in {pwd}:")
        raise

    else:
        log.info(f"Ferre finished")

    return None
