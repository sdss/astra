
# Submit something to notchpeak-gpu just to be able to test stuff

kwds = {
    'alloc': 'notchpeak-gpu',
    'partition': 'notchpeak-gpu',
    'nodes': 1,
    'mem': 16000,
    'walltime': '12:00:00',
    'ppn': 1,
    'gres': 'gpu'
}
from slurm import queue


q = queue(verbose=True)
q.create(label="test-gpu", **kwds)
q.append("sleep 43200")
q.commit(hard=True, submit=True)
