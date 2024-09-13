
from typing import Callable, Union

# For intermediate outputs, there should be an accessor that tries to get it from the real path (eg astraStar file) but if that fails it gets it from the intermediate output path
# and there should be a standard for the intermediate output path (?pickle ?hdf5)
# and the intermediate output path should depend ONLY on spectrum_pk


# task must be able to specify:
# - which types of spectra it takes
# - which type it should prioritise (eg coadd > visit) ?????
# - where clauses for what kind of spectra it takes (eg by color, source, targeting)
# - pre loading function 
# - slurm profile? requirements?
# - how to handle exceptions?
# - write to database?
# - default execution kwds??? have them read from astra.config
# - needs to be able to extend the specrum query to make additional selecton cuts (eg Corv requiring SnowWhite DA-type), or SLAM only running in some parts of param space
# - allow DEBUG as a special kwarg that will re-raise all exceptions 

# TODO: what if the task runs on a SOURCE and not a SPECTRUM?
#       ... I think that is OK, the introspection should just check for the type expected (eg Source, not Spectrum)
#       and then it should distribute the work accordingly


@task(
    pre_process_callable=my_task_pre_loader, # executed once per process
    select_query_callable=lambda q: (
        q
        .join(SnowWhite)
        .where(SnowWhite.classification == "DA")        
    ), # used before distributing work
)
def my_task(spectrum: ApogeeRestFrameSpectrum, **kwargs) -> OutputType:
    """
    Does shit
    """
    context = kwargs.get("pre_execution_callable") # result from pre-loader???

    # write individual file

    return X


# So for something like ASPCAP, what does that look like?
# When the astra_execute thing is run, that wants to run on a whole bunch of spectra
# and to distribute across nodes, etc. So *that* is the point where we need to do load
# balancing and set up FERRE, because by the time it gets to the process level, the
# distribution has already happened.

def post_process_ferre(context, **kwargs):
    pwd = context["pre_distribute_result"]["pwd"]

    return {
        # spectrum_pk -> result kwds
    }

@task(
    pre_distribute_callable=load_balance_and_pre_process_ferre,
    pre_process_callable=post_process_ferre
)
def ferre_initial(spectrum: ApogeeRestFrameSpectrumType, context: dict, **kwargs) -> FerreInitial:

    pre_process_result = context["pre_process_result"]
    kwds = pre_process_result[spectrum.spectrum_pk]

    return FerreInitial(**kwds)


def load_balance_and_pre_process_ferre():

    # the @task will:
    # 1. query for what spectra to run, and how many, etc.
    # 2. if there is a pre_distribute_callable, then it is going to run that on the spectra before setting up slurm jobs etc
    #    -> if there isn't one then the slurm jobs would just be paginating the query out to each process, etc.
    # 3. the pre_distribute_callable should return some SlurmJobs or something like that.
    # 4. those slurm jobs should run ferre_initial on the pwd when they are complete, with the 'pre_distribute_result' providing the pwd
    # 5. the pre_process_callable would process the FERRE run



def design_matrix(path: "some_model.pt", spectrum, **kwargs):
    # spectrum must 
    return A, L, label_names


@task(pre_process_callable=design_matrix)
def whow(spectrum: AnySpectrumType, model_path: str, **kwargs) -> WHOW:

    A, L, label_names = kwargs.get("pre_process_callable_result")

    Y = spectrum.flux
    Cinv = spectrum.ivar
    
    ATCinv = A.T @ Cinv
    X = np.linalg.solve(ATCinv @ A, ATCinv @ Y)

    rchi2 = ...
    labels = L @ X[:32]

    r = WHOW.from_spectrum(
        spectrum,
        **dict(zip(label_names, labels))
    )

    # TODO: need to know whether this spectrum is star-level type or not, right?
    with fits.open(r.absolute_path) as image:
        image.write(...)
    
    return r



distribute(
    my_task, 
    ApogeeCoaddedSpectrumInApStar, 
    where=None, 
    slurm_profile="notchpeak", 
    nodes=4, 
    limit=10_000, 
    threads=32, 
    processes=10
) #-> Slurm jobs?

# steps would be:
# - executor performs a query to get all things not yet analyzed, which could then be paginated to different nodes/procs/etc
# - executor submits slurm jobs across each node/proc/etc
# - within each proc:
#   + executor gets its pages to do
#   + executor does introspection to see there is a pre-execute hook
#   + it runs the pre-execute hook once and creates a context object --> this pre-execute hook must have access to the first spectrum
#   + it directly runs the task function for all spectra, supplying the context object each time
#   + it adds the processor overhead, how many spectra executed in this batch (since it knows hat), and the time for this single analysis
#   + at some intervals, it batch inserts the results to the database, following some rules about what to do if there are integrity conflicts
