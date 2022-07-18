from platform import release
from unittest.mock import NonCallableMagicMock
import numpy as np
import datetime
import json
import os
from astropy.io import fits
from functools import cached_property
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.constants import c
from healpy import ang2pix

from astra import (log, __version__ as astra_version)
from astra.database.astradb import (database, DataProduct, TaskOutputDataProducts)
from astra.utils import flatten, expand_path

from peewee import fn, ForeignKeyField, JOIN, Expression, Alias, Field
from astra.database.astradb import Source, Task, DataProduct, Output, TaskInputDataProducts, SourceDataProduct, TaskOutput
from astra.database.apogee_drpdb import Star, Visit

from astropy.table import Table, MaskedColumn
from sdss_access import SDSSPath
from functools import lru_cache

from typing import List, Tuple, Dict, Union, Optional


from .catalog import (get_sky_position, get_gaia_dr2_photometry)

c_km_s = c.to(u.km / u.s).value


def get_log_lambda_dispersion_kwds(wavelength, decimals=6):
    log_lambda = np.log10(wavelength)
    unique_diffs = np.unique(np.round(np.diff(log_lambda), decimals=decimals))
    if unique_diffs.size > 1:
        raise ValueError(f"Wavelength array is not uniformly sampled in log wavelength: deltas={unique_diffs}")
    return (log_lambda[0], unique_diffs[0], 1)




def create_object_identifier(ra, dec):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ra_str = coord.to_string("hmsdms", sep="", precision=2).split()[0]
    dec_str = coord.to_string("hmsdms", sep="", precision=1).split()[1]
    return (f"2M{ra_str}{dec_str}").replace(".", "")


def get_source_metadata_from_task(task):
    """
    Return a dictionary of metadata for the source associated with a task.
    """
    input_data_products = tuple(task.input_data_products)

    # Get source and catalogid.
    sources = tuple(set(flatten([list(dp.sources) for dp in input_data_products])))
    source, *other_sources = sources
    if len(other_sources) > 0:
        raise ValueError(f"More than one source associated with the input data products: {sources} -> {input_data_products}")
    catalog_id = source.catalogid    

    # Sky position.
    ra, dec = get_sky_position(catalog_id)

    # Object identifier.
    for data_product in input_data_products:
        if "obj" in data_product.kwargs:
            object_id = data_product.kwargs["obj"]
            break
    else:
        log.warning(f"No object identifier found for {source} among {input_data_products}, creating one instead")
        object_id = create_object_identifier(ra, dec)

    # Nside = 128 is fixed in SDSS-V!        
    healpix = ang2pix(128, ra, dec, lonlat=True)

    return dict(
        input_data_products=input_data_products,
        catalog_id=catalog_id,
        ra=ra,
        dec=dec,
        object_id=object_id,
        healpix=healpix
    )



COMMON_OUTPUTS = {
    "output": ("OUTPUTID", "Astra unique output identifier"),
    "task": ("TASKID", "Astra unique task identifier"),
    "meta": ("META", "Spectrum metadata"),
    "snr": ("SNR", "Signal-to-noise ratio"),
    "teff": ("TEFF", "Effective temperature [K]"),
    "logg": ("LOGG", "Log10 surface gravity [dex]"),
    "u_teff": ("U_TEFF", "Uncertainty in effective temperature [K]"),
    "u_logg": ("U_LOGG", "Uncertainty in surface gravity [dex]"),
}

def create_AstraAllStar_product(
    model,
    gzip=True
):
    from peewee import JOIN
    from sdssdb.peewee.sdss5db import database as sdss5_database
    sdss5_database.set_profile("operations") # TODO: HOW CAN WE SET THIS AS DEFAULT!?!

    from sdssdb.peewee.sdss5db.catalogdb import (Catalog, Gaia_DR2, CatalogToTIC_v8, TIC_v8, TwoMassPSC)
    from sdssdb.peewee.sdss5db.targetdb import (Target, CartonToTarget, Carton)



    # Ignore any `meta` columns for now
    fields = list(filter(
        lambda c: not isinstance(c, ForeignKeyField) and c.name != "meta", 
        model._meta.sorted_fields
    ))

    # Select one result per Source (catalogid).
    # If there are multiple tasks that have analysed that source, then get the most recent.
    # If there are multiple outputs associated with that task, then get the first

    q_results = (
        Task.select(
                Source.catalogid,
                DataProduct.release,
                DataProduct.filetype,
                DataProduct.kwargs,
                Task.id.alias("task_id"),
                Task.version,
                Task.time_total,
                Task.created,
                Task.parameters,
                Output.id.alias("output_id"),
                *fields
            )
            .distinct(Source.catalogid)
            .join(TaskInputDataProducts)
            .join(DataProduct)
            .join(SourceDataProduct)
            .join(Source)
            .switch(Task)
            .join(TaskOutput, JOIN.LEFT_OUTER)
            .join(Output)
            .join(model)
            .order_by(Source.catalogid.asc(), Task.id.desc(), Output.id.asc()) # Get the most recent task, but the first output in that task
            .dicts()
    )

    log.debug(f"Querying {q_results}")

    results = []
    catalogids = set()
    parameter_sets = []
    last_kwargs = None
    for N, row in enumerate(q_results, start=1):
        catalogids.add(row["catalogid"])
        if "parameters" in row:
            parameters = row.pop("parameters")
            try:
                parameter_sets.append(frozenset(parameters))
            except:
                log.warning(f"Failed to serialize parameters: {parameters}")
            
        row["created"] = row["created"].isoformat()
        last_kwargs = row.pop("kwargs")
        row.update(last_kwargs)
    
        results.append(row)

    log.info(f"We have {N} result rows")

    log.info(f"Querying carton data for {len(catalogids)} sources")

    sq = (
        Carton.select(
            Target.catalogid, 
            Carton.carton,
            Carton.program
        )
        .distinct()
        .join(CartonToTarget)
        .join(Target)
        .where(Target.catalogid.in_(catalogids))
        .alias("distinct_cartons")
    )

    q_cartons = (
        Target.select(
            Target.catalogid, 
            fn.STRING_AGG(sq.c.carton, ",").alias("cartons"),
            fn.STRING_AGG(sq.c.program, ",").alias("programs"),
        )
        .join(sq, on=(sq.c.catalogid == Target.catalogid))
        .group_by(Target.catalogid)
        .order_by(Target.catalogid.asc())
        .tuples()
    )

    targeting = { cid: (c, p) for cid, c, p in q_cartons }
    log.info(f"Adding carton information ({len(targeting)})..")
    N_missing_targeting_info = 0
    for row in results:
        catalogid = row["catalogid"]
        try:
            cartons, programs = targeting[catalogid]
        except:
            N_missing_targeting_info += 1
            cartons, programs = ("", "")
        row["cartons"] = cartons
        row["programs"] = programs

    if N_missing_targeting_info > 0:
        log.warning(f"There were {N_missing_targeting_info} catalog sources without targeting info (no cartons or programs)")

    log.info(f"Querying photometry on {len(catalogids)} sources from {Gaia_DR2} and {TwoMassPSC}")

    # Supply with metadata from the catalog
    q_meta = (
        Catalog.select(
            Catalog.catalogid,
            Catalog.ra,
            Catalog.dec,
            TIC_v8.id.alias("tic_v8_id"),
            Gaia_DR2.source_id.alias("gaia_dr2_source_id"),
            Gaia_DR2.phot_g_mean_mag,
            Gaia_DR2.phot_bp_mean_mag,
            Gaia_DR2.phot_rp_mean_mag,
            Gaia_DR2.parallax,
            Gaia_DR2.parallax_error,
            Gaia_DR2.pmra,
            Gaia_DR2.pmra_error,
            Gaia_DR2.pmdec,
            Gaia_DR2.pmdec_error,
            Gaia_DR2.bp_rp,
            Gaia_DR2.radial_velocity,
            Gaia_DR2.radial_velocity_error,
            TwoMassPSC.j_m,
            TwoMassPSC.h_m,
            TwoMassPSC.k_m,
        )
        .distinct(Catalog.catalogid)
        .join(CatalogToTIC_v8, JOIN.LEFT_OUTER)
        .join(TIC_v8)
        .join(Gaia_DR2, JOIN.LEFT_OUTER)
        .switch(TIC_v8)
        .join(TwoMassPSC, JOIN.LEFT_OUTER)
        .where(Catalog.catalogid.in_(list(catalogids)))
        .order_by(Catalog.catalogid.asc())
        .dicts()
    )

    meta = { row["catalogid"]: row for row in q_meta}
    missing_metadata = []

    names = []
    ignore_names = ("parameters", )
    for query in (q_meta, q_cartons, q_results):
        for field in query._returning:
            if isinstance(field, Expression):
                name = field.rhs
            elif isinstance(field, Field):
                name = field.name
            elif isinstance(field, Alias):
                name = field._alias
            else:
                raise RuntimeError(f"Cannot get name for field type ({type(field)} ({field}) of {query}")

            if name == "kwargs":
                for name in last_kwargs.keys():
                    if name not in names:
                        names.append(name)
            else:
                if name not in names and name not in ignore_names:
                    if name not in names:
                        names.append(name)

    default_values_by_type = {
        str: "",
        float: np.nan,
        int: -1,
    }

    try:
        first_key, *_ = meta.keys()
        no_meta = { k: default_values_by_type[type(v)] for k, v in meta[first_key].items() }
    except:
        no_meta = {}
        
    M, C = (0, 0)
    for row in results:
        catalogid = row["catalogid"]
        try:
            row.update(meta[catalogid])
        except KeyError:
            if catalogid not in missing_metadata:
                log.warning(f"No metadata found for catalogid {catalogid}!")
                missing_metadata.append(catalogid)
                for key, value in no_meta.items():
                    row.setdefault(key, value)
                
            M += 1
        else:
            C += 1

    if len(missing_metadata) > 0:
        log.warning(f"In total there are {len(missing_metadata)} catalog sources without metadata!")

    # If NONE of the rows have metadata, then we will get an error when we try to build a table.
    # We should fill the first result with empty values.
    if C == 0:
        missing_field_names = set(names).difference(results[0])
        results[0].update({ name: np.nan for name in missing_field_names })
        log.warning(f"ALL sources are missing metadata!")
    
    # If we are combining results from multiple data products, then the columns will be different.
    # We should find incomplete columns and a default value (type) for each.


    try:
        table = Table(
            data=results,
            names=names
        )
    except ValueError:
        log.exception(f"Exception when first creating table. Trying to fill in missing values.")
        sometimes_missing = []
        missing_types = {}
        for row in results:
            for key in set(names).difference(row):#set(row).difference(names):
                if key in sometimes_missing: continue

                sometimes_missing.append(key)

            for key in set(sometimes_missing).intersection(row).difference(missing_types):
                missing_type = type(row[key])
                missing_types[key] = default_values_by_type[missing_type]

        log.info(f"Adding defaults for these columns: {missing_types}")
        for row in results:
            for key in set(missing_types).difference(row):
                row[key] = missing_types[key]

        for key in missing_types.keys():
            names.insert(names.index("task_id"), key)

    table = Table(
        data=results,
        names=names,            
    )
    # Fix dtypes etc.
    fill_values = {
        float: np.nan,
        int: -1
    }
    for index, (name, dtype) in enumerate(table.dtype.descr):
        if dtype == "|O":
            # Objects.
            mask = (table[name] == None)
            if all(mask) or (not any(mask) and len(set(table[name])) == 1):
                # All Nones, probably. Delete.
                del table[name]
            else:
                data = np.array(table[name])
                del table[name]

                kwds = dict(name=name)
                if any(mask):
                    dtype = type(data[~mask][0])
                    fill_value = fill_values[dtype]
                    data[mask] = fill_value
                    kwds.update(
                        mask=mask,
                        dtype=dtype,
                        fill_value=fill_value
                    )

                table.add_column(
                    MaskedColumn(data, **kwds),
                    index=index
                )

    # e.g., ApogeeNetOutput
    component_name = model.__name__[:-len("Output")]

    path = expand_path(f"$MWM_ASTRA/{astra_version}/astraAllStar-{component_name}-{astra_version}.fits")
    if gzip: path += ".gz"
    table.write(path, overwrite=True)

    log.info(f"Created file: {path}")
    return path


# TODO: Refactor this to something that can be used by astra/operators/sdss and here.
def get_boss_visits(catalogid):
    data = Table.read(expand_path("$BOSS_SPECTRO_REDUX/master/spAll-master.fits"))
    matches = np.where(data["CATALOGID"] == catalogid)[0]

    kwds = []
    for row in data[matches]:
        kwds.append(dict(
            # TODO: remove this when the path is fixed in sdss_access
            fieldid=f"{row['FIELD']:0>6.0f}",
            mjd=int(row["MJD"]),
            catalogid=int(catalogid),
            run2d=row["RUN2D"],
            isplate=""
        ))
    return kwds

# TODO: Refactor this to something that can be used by astra/operators/sdss and here.

@lru_cache
def path_instance(release):
    return SDSSPath(release=release)

@lru_cache
def lookup_keys(release, filetype):
    return path_instance(release).lookup_keys(filetype)


def get_apogee_visits(catalogid, release="sdss5"):
    q = (
        Visit.select()
             .where(Visit.catalogid == catalogid)
    )
    kwds = []
    for row in q:
        kwds.append(
            { k: getattr(row, k) for k in lookup_keys(release, "apVisit") }
        )
    return kwds




def create_mwmVisits_product(
    catalog_id: int,
    overwrite=True,
    release="sdss5",
) -> DataProduct:
    """
    Create a data product that contains all visit spectra for a source.
    """

    # Sky position.
    ra, dec = get_sky_position(catalog_id)
    healpix = ang2pix(128, ra, dec, lonlat=True)

    # Get all visits.
    visits = get_apogee_visits(catalog_id, release) \
           + get_boss_visits(catalog_id)

    # Object identifier.
    for kwds in visits:
        object_id = kwds.get("obj", None)
        if object_id is not None:
            break
    else:
        object_id = create_object_identifier(ra, dec)

    # Primary FITS header
    hdu_primary = fits.PrimaryHDU(
        header=fits.Header(
            [
                (
                    "DATE", 
                    datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                    "File creation date (UTC)"
                ),
                (   
                    "ASTRAVER", 
                    astra_version,
                    "Software version of Astra"
                ),
                (
                    "CATID", 
                    catalog_id,
                    "SDSS-V catalog identifier"
                ),
                (
                    "OBJID",
                    object_id,
                    "Object identifier"
                ),
                (
                    "RA",
                    ra,
                    "RA (J2000)"
                ),
                (
                    "DEC",
                    dec,
                    "DEC (J2000)" 
                ),
                (
                    "HEALPIX",
                    healpix,
                    "HEALPix location"
                ),
            ]
        )
    )
    # TODO: Add comments about what is in each HDU.    


    raise NotImplementedError("Not implemented yet.")


    path = expand_path(
        f"$MWM_ASTRA/{astra_version}/healpix/{healpix // 1000}/{healpix}/"
        f"mwmVisits-{apred_version}-{boss_version}-{catalogid}.fits"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)


from scipy import interpolate
from scipy.ndimage.filters import median_filter, gaussian_filter


def wave2pix(wave,wave0) :
    """ convert wavelength to pixel given wavelength array
    Args :
       wave(s) : wavelength(s) (\AA) to get pixel of
       wave0 : array with wavelength as a function of pixel number 
    Returns :
       pixel(s) in the chip
    """
    pix0= np.arange(len(wave0))
    # Need to sort into ascending order
    sindx= np.argsort(wave0)
    wave0= wave0[sindx]
    pix0= pix0[sindx]
    # Start from a linear baseline
    baseline= np.polynomial.Polynomial.fit(wave0,pix0,1)
    ip= interpolate.InterpolatedUnivariateSpline(wave0,pix0/baseline(wave0),k=3)
    out= baseline(wave)*ip(wave)
    # NaN for out of bounds
    out[wave > wave0[-1]]= np.nan
    out[wave < wave0[0]]= np.nan
    return out


def sincint(x, nres, speclist) :
    """ Use sinc interpolation to get resampled values
        x : desired positions
        nres : number of pixels per resolution element (2=Nyquist)
        speclist : list of [quantity, variance] pairs (variance can be None)
    """

    dampfac = 3.25*nres/2.
    ksize = int(21*nres/2.)
    if ksize%2 == 0 : ksize +=1
    nhalf = ksize//2 

    #number of output and input pixels
    nx = len(x)
    nf = len(speclist[0][0])

    # integer and fractional pixel location of each output pixel
    ix = x.astype(int)
    fx = x-ix

    # outputs
    outlist=[]
    for spec in speclist :
        if spec[1] is None :
            outlist.append([np.full_like(x,0),None])
        else :
            outlist.append([np.full_like(x,0),np.full_like(x,0)])

    for i in range(len(x)) :
        xkernel = np.arange(ksize)-nhalf - fx[i]
        # in units of Nyquist
        xkernel /= (nres/2.)
        u1 = xkernel/dampfac
        u2 = np.pi*xkernel
        sinc = np.exp(-(u1**2)) * np.sin(u2) / u2
        sinc /= (nres/2.)

        lobe = np.arange(ksize) - nhalf + ix[i]
        vals = np.zeros(ksize)
        vars = np.zeros(ksize)
        gd = np.where( (lobe>=0) & (lobe<nf) )[0]

        for spec,out in zip(speclist,outlist) :
            vals = spec[0][lobe[gd]]
            out[0][i] = (sinc[gd]*vals).sum()
            if spec[1] is not None : 
                var = spec[1][lobe[gd]]
                out[1][i] = (sinc[gd]**2*var).sum()

    for out in outlist :
       if out[1] is not None : out[1] = np.sqrt(out[1])
    
    return outlist

from astra.sdss.apogee_bitmask import PixelBitMask

def combine_boss_visits(
    visits,
    release="sdss5",
    nres=5,
    min_mean_visit_snr=5,
):
    """
    Combine BOSS spectra of the same source to a single rest-frame spectrum.
    """

    crval, cdelt, crpix, n_pixels = (3.5523, 1e-4, 1, 4648)
    resampled_wavelength = 10**(crval + cdelt * np.arange(n_pixels))
    n_visits = len(visits)

    shape = (n_visits, n_pixels)
    resampled_flux = np.zeros(shape)
    resampled_flux_error = np.zeros(shape)
    resampled_pseudo_cont = np.zeros(shape)
    resampled_sky_flux = np.zeros(shape)
    resampled_bitmask_or = np.zeros(shape, dtype=int)
    resampled_bitmask_and = np.zeros(shape, dtype=int)
    redshift = np.zeros(n_visits, dtype=float)

    for i, kwds in enumerate(visits):
        path = path_instance(release).full("specLite", **kwds)
        if not os.path.exists(path):
            log.warning(f"Missing file: {path} from {kwds}")
            continue

        with fits.open(path) as image:

            z, = image[2].data["Z"] # TODO: Do we want -z here?
            redshift[i] = z
            visit_rest_wavelength = resampled_wavelength * (1.0 + z)

            flux = image[1].data["FLUX"]
            flux_error = image[1].data["IVAR"]**-0.5

            pix = wave2pix(visit_rest_wavelength, 10**image[1].data["LOGLAM"])
            gd, = np.where(np.isfinite(pix))

            # Get a smoothed, filtered spectrum to use as replacement for bad values
            '''
            cont = gaussian_filter(
                median_filter(
                    flux,
                    [501],
                    mode='reflect'
                ),
                100
            )
            cont_error = gaussian_filter(
                median_filter(
                    flux_error
                    [501],
                    mode='reflect'
                ),
                100
            )


            bd, = np.where(image[hdu_bitmask].data[chip] & pixelmask.badval())
            if len(bd) > 0: 
                chip_flux[bd] = cont[bd] 
                chip_flux_error[bd] = cont_error[bd] 
            '''

            # Do the sinc interpolation
            raw = [
                [flux, flux_error], # flux
                [image[1].data["SKY"], None], # sky
            ]
            # Load up individual mask bits
            #for ibit,name in enumerate(pixelmask.name):
            #    if name != '' and len(np.where(image[hdu_bitmask].data[chip] & 2**ibit)[0]) > 0:
            #        raw.append([np.clip(image[hdu_bitmask].data[chip] & 2**ibit, None, 1), None])

            out = sincint(pix[gd], nres, raw)

            resampled_flux[i, gd], resampled_flux_error[i, gd] = out[0]
            resampled_sky_flux[i, gd], _ = out[1]

            # From output flux, get continuum to remove, so that all spectra are
            #   on same scale. We'll later multiply in the median continuum
            resampled_pseudo_cont[i, gd] = gaussian_filter(
                median_filter(
                    resampled_flux[i, gd],
                    [501],
                    mode='reflect'
                ),
                100
            )
            resampled_flux[i, gd] /= resampled_pseudo_cont[i, gd]
            resampled_flux_error[i, gd] /= resampled_pseudo_cont[i, gd]

            '''
            # For mask, set bits where interpolated value is above some threshold
            # defined for each mask bit
            iout = 3
            for ibit,name in enumerate(pixelmask.name):
                if name != '' and len(np.where(image[hdu_bitmask].data[chip] & 2**ibit)[0]) > 0:
                    j = np.where(np.abs(out[iout][0]) > pixelmask.maskcontrib[ibit])[0]
                    resampled_bitmask[i, gd[j]] |= 2**ibit
                    iout += 1
            '''



    # Pixel-by-pixel weighted average
    resampled_ivar = 1.0 / resampled_flux_error**2
    resampled_ivar[resampled_flux_error == 0] = 0

    pixel_snr_clipped = np.clip(resampled_flux * np.sqrt(resampled_ivar), 0, np.inf)
    estimate_snr = np.mean(pixel_snr_clipped, axis=1)
    if min_mean_visit_snr is not None:
        keep = estimate_snr >= min_mean_visit_snr
    else:
        keep = np.ones(estimate_snr.size, dtype=bool)

    cont = np.median(resampled_pseudo_cont[keep], axis=0) # TODO: is this right?
    stacked_ivar = np.sum(resampled_ivar[keep], axis=0)
    stacked_flux = np.sum(resampled_flux[keep] * resampled_ivar[keep], axis=0) / stacked_ivar * cont
    stacked_flux_error = np.sqrt(1.0/stacked_ivar) * cont

    #stacked_flux = np.sum(resampled_flux/resampled_flux_error**2,axis=0)\
    #                / np.sum(resampled_flux_error**(-2),axis=0) * cont

    #stacked_flux = np.nansum(resampled_flux/resampled_flux_error**2,axis=0)\
    #             / np.nansum(resampled_flux_error**(-2),axis=0) * cont
    #stacked_flux_error =  np.sqrt(1./np.sum(1./resampled_flux_error**2,axis=0)) * cont
    #stacked_bitmask = np.bitwise_and.reduce(resampled_bitmask, 0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(resampled_wavelength, stacked_flux, c='k')
    fig.savefig("tmp.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 12))
    #for i, is_keep in range(resampled_flux.shape[0]):
    for i, is_keep in enumerate(keep):
        ax.plot(resampled_wavelength, resampled_flux[i] + i, c="#666666" if is_keep else "tab:red",
            zorder=1 if is_keep else -1
        )

    ax.plot(resampled_wavelength, stacked_flux / cont + keep.size, c="k")
        
    ax.set_ylim(0, keep.size + 3)
    ax.set_xlim(8520, 8660)
    ax.axvline(8542, c="tab:red")

    fig.savefig("boss2.png")

    


    raise a



'''
from astra.sdss.datamodels import combine_boss_visits, get_boss_visits

catalogid = 27021597917837494
visits = get_boss_visits(catalogid)
combine_boss_visits(visits)
 [14]: Counter(data[ok]["CATALOGID"]).most_common(10)
Out[14]: 
[(27021597917837494, 13),
 (27021597917837548, 13),
 (27021597917837925, 13),
 (27021597917838188, 13),
 (27021597917838529, 13),
 (27021597917838783, 13),
 (27021597918264193, 13),
 (27021597918264237, 13),
 (27021597918264389, 13),


'''



def combine_apogee_visits(
    visits: List[Visit], 
    nres=[5, 4.25, 3.5], # number of pixels per resolution element?
    bc_only=False,
    release="sdss5", 
    filetype="apVisit"
):
    """
    Combine APOGEE visits of the same source to a single rest-frame spectrum.
    """

    # This broadly duplicates what apogee_drp.apred.rv.visitcomb does.

    # Define the wavelength sampling to use.
    crval, cdelt, crpix, n_pixels = (4.179, 6.0e-6, 1, 8575)
    resampled_wavelength = 10**(crval + cdelt * np.arange(n_pixels))
    n_visits = len(visits)

    shape = (n_visits, n_pixels)
    resampled_flux = np.zeros(shape)
    resampled_flux_error = np.zeros(shape)
    resampled_sky_flux = np.zeros(shape)
    resampled_sky_flux_error = np.zeros(shape)
    resampled_telluric_flux = np.zeros(shape)
    resampled_telluric_flux_error = np.zeros(shape)
    resampled_pseudo_cont = np.zeros(shape)
    resampled_bitmask = np.zeros(shape, dtype=int)

    pixelmask = PixelBitMask()

    for i, visit in enumerate(visits):
        kwds = { k: getattr(visit, k) for k in lookup_keys(release, filetype) }
        path = path_instance(release).full(filetype, **kwds)
        if not os.path.exists(path):
            log.warning(f"Missing {filetype} file: {path} from {kwds}")
            continue

        with fits.open(path) as image:
            '''
            HISTORY AP1DVISIT:  HDU0 = Header only
            HISTORY AP1DVISIT:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)
            HISTORY AP1DVISIT:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)
            HISTORY AP1DVISIT:  HDU3 - Flag mask (bitwise OR combined)
            HISTORY AP1DVISIT:  HDU4 - Wavelength (Ang)
            HISTORY AP1DVISIT:  HDU5 - Sky (10^-17 ergs/s/cm^2/Ang)
            HISTORY AP1DVISIT:  HDU6 - Sky Error (10^-17 ergs/s/cm^2/Ang)
            HISTORY AP1DVISIT:  HDU7 - Telluric
            HISTORY AP1DVISIT:  HDU8 - Telluric Error
            HISTORY AP1DVISIT:  HDU9 - Wavelength coefficients
            HISTORY AP1DVISIT:  HDU10 - LSF coefficients
            '''
            hdu_header, hdu_flux, hdu_flux_error, hdu_bitmask, hdu_wl, \
                hdu_sky, hdu_sky_error, \
                hdu_telluric, hdu_telluric_error, \
                hdu_wl_coeff, hdu_lsf_coeff = range(11)
            
            v_rel = -visit.bc if bc_only else visit.vrel

            if not np.isfinite(v_rel):
                # TODO: could require finite vrel in the initial query..
                log.warning(f"Non-finite velocity for {path}. Skipping..")
                continue
        
            visit_rest_wavelength = resampled_wavelength * (1.0 + v_rel / c_km_s)

            for chip, chip_nres in enumerate(nres):

                chip_flux = image[hdu_flux].data[chip]
                chip_flux_error = image[hdu_flux_error].data[chip]


                pix = wave2pix(visit_rest_wavelength, image[hdu_wl].data[chip])
                gd, = np.where(np.isfinite(pix))

                # Get a smoothed, filtered spectrum to use as replacement for bad values
                cont = gaussian_filter(
                    median_filter(
                        image[hdu_flux].data[chip],
                        [501],
                        mode='reflect'
                    ),
                    100
                )
                cont_error = gaussian_filter(
                    median_filter(
                        # TODO: The apogee_drp uses flux here, but I think they mean to use flux error.
                        image[hdu_flux_error].data[chip],
                        [501],
                        mode='reflect'
                    ),
                    100
                )
            
                bd, = np.where(image[hdu_bitmask].data[chip] & pixelmask.badval())
                if len(bd) > 0: 
                    chip_flux[bd] = cont[bd] 
                    chip_flux_error[bd] = cont_error[bd] 

                # Do the sinc interpolation
                raw = [
                    [chip_flux, chip_flux_error], # flux
                    [image[hdu_sky].data[chip], image[hdu_sky_error].data[chip]**2], # sky
                    [image[hdu_telluric].data[chip], image[hdu_telluric_error].data[chip]**2] # telluric
                ]
                # Load up individual mask bits
                for ibit,name in enumerate(pixelmask.name):
                    if name != '' and len(np.where(image[hdu_bitmask].data[chip] & 2**ibit)[0]) > 0:
                        raw.append([np.clip(image[hdu_bitmask].data[chip] & 2**ibit, None, 1), None])

                out = sincint(pix[gd], chip_nres, raw)

                resampled_flux[i, gd], resampled_flux_error[i, gd] = out[0]
                resampled_sky_flux[i, gd], resampled_sky_flux_error[i, gd] = out[1]
                resampled_telluric_flux[i, gd], resampled_telluric_flux_error[i, gd] = out[2]

                 # From output flux, get continuum to remove, so that all spectra are
                #   on same scale. We'll later multiply in the median continuum
                resampled_pseudo_cont[i, gd] = gaussian_filter(
                    median_filter(
                        resampled_flux[i, gd],
                        [501],
                        mode='reflect'
                    ),
                    100
                )
                resampled_flux[i, gd] /= resampled_pseudo_cont[i, gd]
                resampled_flux_error[i, gd] /= resampled_pseudo_cont[i, gd]

                # For mask, set bits where interpolated value is above some threshold
                # defined for each mask bit
                iout = 3
                for ibit,name in enumerate(pixelmask.name):
                    if name != '' and len(np.where(image[hdu_bitmask].data[chip] & 2**ibit)[0]) > 0:
                        j = np.where(np.abs(out[iout][0]) > pixelmask.maskcontrib[ibit])[0]
                        resampled_bitmask[i, gd[j]] |= 2**ibit
                        iout += 1


        # Increase uncertainties for persistence pixels
        bd, = np.where((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_HIGH')) > 0)
        if len(bd) > 0: resampled_flux_error[i,bd] *= np.sqrt(5)
        bd, = np.where(((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_HIGH')) == 0) &
                       ((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_MED')) > 0) )
        if len(bd) > 0: resampled_flux_error[i,bd] *= np.sqrt(4)
        bd, = np.where(((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_HIGH')) == 0) &
                       ((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_MED')) == 0) &
                       ((resampled_bitmask[i,:] & pixelmask.getval('PERSIST_LOW')) > 0) )
        if len(bd) > 0: resampled_flux_error[i,bd] *= np.sqrt(3)
        bd, = np.where((resampled_bitmask[i,:] & pixelmask.getval('SIG_SKYLINE')) > 0)
        if len(bd) > 0: resampled_flux_error[i,bd] *= np.sqrt(100)

    # Pixel-by-pixel weighted average
    cont = np.median(resampled_pseudo_cont, axis=0)
    stacked_flux = np.sum(resampled_flux/resampled_flux_error**2,axis=0)\
                    / np.sum(resampled_flux_error**(-2),axis=0) * cont
    stacked_flux_error =  np.sqrt(1./np.sum(1./resampled_flux_error**2,axis=0)) * cont
    stacked_bitmask = np.bitwise_and.reduce(resampled_bitmask, 0)
    raise a


'''
from astra.sdss.datamodels import combine_apogee_visits
from astra.database.apogee_drpdb import Visit, RvVisit
from peewee import fn

visits = (
    RvVisit.select(
            RvVisit,
            Visit
        )
        .distinct(RvVisit.visit_pk)
        .join(Visit, on=(Visit.pk == RvVisit.visit_pk))
        .where(Visit.catalogid == 4375918863)
        .order_by(RvVisit.visit_pk.desc(), RvVisit.pk.desc())
        .objects()
)

stack = combine_apogee_visits(visits)




'''


def create_AstraStar_product(
    task,
    *related_tasks,
    wavelength=None,
    model_flux=None,
    model_ivar=None,
    rectified_flux=None,
    crval=None,
    cdelt=None,
    crpix=None,
    overwrite=True,
    release="sdss5",
    **kwargs
):
    """
    Create an AstraStar data product that contains output parameters and best-fit model(s) from one
    pipeline for a single star.
    
    :param task:
        The primary task responsible for creating this data product.
    
    :param related_tasks: [optional]
        Any related tasks whose parameters should be stored with this file.
    
    :param model_flux: [optional]
        The best-fitting model flux.
    """

    meta = get_source_metadata_from_task(task)
    catalog_id, healpix = (meta["catalog_id"], meta["healpix"]) # need these later for path

    hdu_primary = fits.PrimaryHDU(
        header=fits.Header(
            [
                (
                    "DATE", 
                    datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                    "File creation date (UTC)"
                ),
                (   
                    "ASTRAVER", 
                    astra_version,
                    "Software version of Astra"
                ),
                (
                    "CATID", 
                    catalog_id,
                    "SDSS-V catalog identifier"
                ),
                (
                    "OBJID",
                    meta["object_id"],
                    "Object identifier"
                ),
                (
                    "RA",
                    meta["ra"],
                    "RA (J2000)"
                ),
                (
                    "DEC",
                    meta["dec"],
                    "DEC (J2000)" 
                ),
                (
                    "HEALPIX",
                    healpix,
                    "HEALPix location"
                ),
                (
                    "INPUTS",
                    ",".join([dp.path for dp in meta["input_data_products"]]),
                    "Input data products"
                )
            ]
        )
    )

    # astra.contrib.XXXX.
    component_name = task.name.split(".")[2]

    if wavelength is not None:
        if crval is None and cdelt is None and crpix is None:
            # Figure out these values ourselves.
            crval, cdelt, crpix = get_log_lambda_dispersion_kwds(wavelength)
        else:
            raise ValueError("Wavelength given AND crval, cdelt, crpix")
    

    cards = [
        ("PIPELINE", f"{component_name}", "Analysis component name"),
        ("CRVAL1", crval),
        ("CDELT1", cdelt),
        ("CRPIX1", crpix),
        ("CTYPE1", "LOG-LINEAR"),
        ("DC-FLAG", 1),            
    ]
    # Add results from the task's *first* output only.
    # TODO: What if there are many outputs? I guess we just don't allow that with this data model.
    try:
        output, *_ = task.outputs 
    except:
        log.warning(f"No summary outputs found for task {task}")
    else:
        for k, value in output.__data__.items():
            header_key, description = COMMON_OUTPUTS.get(k, (k.upper(), None))
            cards.append(
                (header_key, value, description)
            )

    header = fits.Header(cards)

    flux_col = fits.Column(name="model_flux", format="E", array=model_flux)
    ivar_col = fits.Column(name="model_ivar", format="E", array=model_ivar)
    rectified_flux_col = fits.Column(name="rectified_flux", format="E", array=rectified_flux)

    hdu_spectrum = fits.BinTableHDU.from_columns(
        [flux_col, ivar_col, rectified_flux_col],
        header=header
    )

    # Task parameters
    all_tasks = [task] + list(related_tasks)
    task_ids = [task.id for task in all_tasks]
    task_names = [task.name for task in all_tasks]
    task_columns = [
        fits.Column(name="task_id", format=get_fits_format_code(task_ids), array=task_ids),
        fits.Column(name="task_name", format=get_fits_format_code(task_names), array=task_names)
    ]

    task_parameters = {}
    for task in all_tasks:
        for parameter_name, parameter_value in task.parameters.items():
            task_parameters.setdefault(parameter_name, [])
            # TODO: Is there a better way to handle Nones? It would be nice if we could 
            #       take the parameters in the fits file and immediately reconstruct a
            #       task, but that's not possible if "" and None mean different things.

            if parameter_value is None:
                parameter_value = ""
            elif isinstance(parameter_value, (dict, list, tuple)):
                parameter_value = json.dumps(parameter_value)
            task_parameters[parameter_name].append(parameter_value)
        
    
    for parameter_name, values in task_parameters.items():
        task_columns.append(
            fits.Column(
                name=parameter_name,
                format=get_fits_format_code(values),
                array=values
            )
        )

    # Store the outputs from each task.
    hdu_tasks = fits.BinTableHDU.from_columns(task_columns)

    # Results in the hdu_tasks?
    # We can use Table(.., descriptions=(..., )) to give comments for each column type
    hdu_list = fits.HDUList([
        hdu_primary,
        hdu_spectrum,
        hdu_tasks
    ])
    # Add checksums to each
    for hdu in hdu_list:
        hdu.add_datasum()
        hdu.add_checksum()

    # TODO: Make this use sdss_access when the path is defined!
    path = expand_path(
        f"$MWM_ASTRA/{astra_version}/healpix/{healpix // 1000}/{healpix}/"
        f"astraStar-{component_name}-{catalog_id}.fits"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu_list.writeto(path, overwrite=overwrite)

    # Create the data product record.
    # TODO: Make this use "AstraStar" filetype when it is defined!
    with database.atomic():
        output_dp, was_created = DataProduct.get_or_create(
            release=release,
            filetype="full",
            kwargs=dict(
                full=path
            )
        )
        TaskOutputDataProducts.create(
            task=task,
            data_product=output_dp
        )
    
    log.info(f"{'Created' if was_created else 'Retrieved'} data product {output_dp} for {task}: {path}")

    return output_dp




def get_fits_format_code(values):
    fits_format_code = {
        bool: "L",
        int: "K",
        str: "A",
        float: "E",
        type(None): "A"
    }.get(type(values[0]))
    assert fits_format_code is not None
    if fits_format_code == "A":
        max_len = max(1, max(map(len, values)))
        return f"{max_len}A"
    return fits_format_code

