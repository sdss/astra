import os
import numpy as np
import datetime
import h5py as h5
from concurrent.futures import ProcessPoolExecutor, as_completed
from peewee import DeferredForeignKey, fn
from playhouse.hybrid import hybrid_property
from astra.fields import (
    AutoField, FloatField, BooleanField, DateTimeField, BigIntegerField, IntegerField, TextField,
    ForeignKeyField, PixelArray, BitField, LogLambdaArrayAccessor
)
from astra.models.base import BaseModel
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.source import Source
from astra.fields import BasePixelArrayAccessor


from astropy.constants import c
from astropy import units as u

C_KM_S = c.to(u.km / u.s).value

def _get_array(dir, key, index):
    with h5.File(f"{dir}/apMADGICS_out_{key}.h5", "r") as fp:
        return fp[key][index]


def _get_arrays(dir, key, indices):
    """
    Like `_get_array`, but reads many rows from one `apMADGICS_out_{key}.h5` dataset with a
    single file open, instead of one open per row. Used by `prefetch_pixel_arrays`, which
    parallelizes calls to this across a process pool: reading a scattered row from these
    files costs ~15-20ms regardless of indexing method (measured -- it's a network
    filesystem latency cost, not a CPU cost), so the win comes from overlapping many of
    those waits across worker processes, not from how a single call reads its own rows.
    """
    with h5.File(f"{dir}/apMADGICS_out_{key}.h5", "r") as fp:
        ds = fp[key]
        return np.array([ds[int(i)] for i in indices])


def _read_chunk(component_dir, key, indices):
    """Worker function for `prefetch_pixel_arrays`: read one chunk of rows for one file."""
    return (component_dir, key), dict(zip(indices, _get_arrays(component_dir, key, indices)))


class ARJLRawKeyAccessor(BasePixelArrayAccessor):

    """
    Base class for ARJL visit accessors that compute a derived pixel array from one or more
    raw `apMADGICS_out_*.h5` datasets. Subclasses set `raw_keys` (the dataset names to read)
    and `_compute` (how to combine them into the accessor's value); `prefetch_pixel_arrays`
    below relies on those same two attributes to batch-read many visits at once.
    """

    raw_keys = ()

    @staticmethod
    def _compute(*raw_arrays):
        raise NotImplementedError

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                raw_arrays = [
                    _get_array(instance.component_dir, key, instance.row_index)
                    for key in self.raw_keys
                ]
                value = self._compute(*raw_arrays)
                if self.transform is not None:
                    value = self.transform(value, None, instance)
                instance.__pixel_data__.setdefault(self.name, value[125:])

            finally:
                return instance.__pixel_data__[self.name]

        return self.field


class ARJLFluxAccessor(ARJLRawKeyAccessor):

    """A class to access ARJL pixel-based arrays."""

    raw_keys = ("x_starLines_v0", "x_residuals_v0", "x_starContinuum_v0")

    @staticmethod
    def _compute(x_starLines_v0, x_residuals_v0, x_starContinuum_v0):
        return 1 + x_starLines_v0 + x_residuals_v0 / x_starContinuum_v0


class ARJLStarLinesFluxAccessor(ARJLRawKeyAccessor):

    """A class to access ARJL pixel-based arrays (1 + x_starLines_v0, without the residuals term)."""

    raw_keys = ("x_starLines_v0",)

    @staticmethod
    def _compute(x_starLines_v0):
        return 1 + x_starLines_v0


class ARJLStarLinesInverseVarianceAccessor(ARJLRawKeyAccessor):

    """A class to access ARJL pixel-based arrays (posterior uncertainty of x_starLines_v0)."""

    raw_keys = ("x_starLines_err_v0",)

    @staticmethod
    def _compute(x_starLines_err_v0):
        return 1 / x_starLines_err_v0**2


class ARJLPixelFlagsAccessor(ARJLRawKeyAccessor):

    """A class to access ARJL pixel-based arrays."""

    raw_keys = ("finalmsk",)

    @staticmethod
    def _compute(finalmsk):
        return finalmsk


class ARJLInverseVarianceAccessor(ARJLRawKeyAccessor):

    """A class to access ARJL pixel-based arrays."""

    raw_keys = ("fluxerr2", "x_starContinuum_v0")

    @staticmethod
    def _compute(fluxerr2, x_starContinuum_v0):
        return (1 / fluxerr2) * x_starContinuum_v0**2


def prefetch_pixel_arrays(visits, field_names=("flux", "ivar", "pixel_flags"), n_workers=8):
    """
    Batch-load pixel array fields (e.g. flux/ivar/pixel_flags) for many ARJL visit
    instances at once.

    Each `ARJLRawKeyAccessor` field normally opens its underlying `apMADGICS_out_*.h5`
    file(s) fresh for every single visit it's accessed on. That's wasteful for something
    like `build_arjl_coadds`, which touches every visit of every source, in two ways: it
    opens the same handful of files thousands of times, and each row it reads is a cold,
    scattered read against a multi-hundred-GB file over a network filesystem (~15-20ms/row,
    measured -- a latency cost, not a CPU one, so a single process reading rows one at a
    time can't go faster than that regardless of how it indexes).

    This instead opens each underlying (component_dir, key) file once, splits its needed
    row indices into chunks, and reads those chunks concurrently across `n_workers` worker
    *processes* (threads don't help here -- h5py serializes all HDF5 calls within a process
    behind a global lock, so concurrent reads only actually overlap across separate
    processes). Results are then used to populate each visit's `__pixel_data__` cache
    directly, so the normal `visit.flux` / `.ivar` / `.pixel_flags` access afterwards is
    just a dict lookup.

    :param visits:
        A list of instances of a single ARJL visit spectrum model.

    :param field_names: [optional]
        The pixel array field names to prefetch (default: flux, ivar, pixel_flags).

    :param n_workers: [optional]
        Number of worker processes to read with concurrently (default: 8). This is a
        shared filesystem -- keep this modest rather than maxing out cores.
    """
    if not visits:
        return

    model = type(visits[0])
    accessors = {name: model._meta.pixel_fields[name] for name in field_names}

    # (component_dir, key) -> row indices needed
    needed = {}
    for accessor in accessors.values():
        for key in accessor.raw_keys:
            for visit in visits:
                needed.setdefault((visit.component_dir, key), set()).add(visit.row_index)

    chunks = []
    for (component_dir, key), indices in needed.items():
        indices = sorted(indices)
        chunk_size = max(1, (len(indices) + n_workers - 1) // n_workers)
        for i in range(0, len(indices), chunk_size):
            chunks.append((component_dir, key, indices[i:i + chunk_size]))

    raw_cache = {file_key: {} for file_key in needed}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_read_chunk, *chunk) for chunk in chunks]
        for future in as_completed(futures):
            file_key, values = future.result()
            raw_cache[file_key].update(values)

    for name, accessor in accessors.items():
        for visit in visits:
            accessor._initialise_pixel_array(visit)
            if name in visit.__pixel_data__:
                continue
            raw_arrays = [
                raw_cache[(visit.component_dir, key)][visit.row_index]
                for key in accessor.raw_keys
            ]
            value = accessor._compute(*raw_arrays)
            if accessor.transform is not None:
                value = accessor.transform(value, None, visit)
            visit.__pixel_data__[name] = value[125:]


class ARJLVisitSpectrum(BaseModel, SpectrumMixin):

    """An ApogeeReduction.jl-reduced visit spectrum."""

    pk = AutoField()

    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        column_name="spectrum_pk"
    )
    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source,
        # We want to allow for spectra to be unassociated with a source so that
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True,
        index=True,
        column_name="source_pk",
        backref="arjl_visit_spectra",
    )

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    catalogid = BigIntegerField(index=True, null=True)
    sdss_id = BigIntegerField(index=True, null=True)

    component_dir = TextField(null=False)
    row_index = IntegerField(index=True, null=False)
    v_arjl = TextField(null=True)

    #> Data Product Keywords
    release = TextField(index=True)
    plate = TextField(index=True)
    telescope = TextField(index=True)
    fiber = IntegerField(index=True)
    mjd = IntegerField(index=True)
    field = TextField(index=True)
    obj = TextField(null=True)
    adjfiberindx = IntegerField()

    #> Radial Velocities
    v_rad = FloatField()
    v_rel = FloatField()
    v_rad_flags = BitField(default=0)
    v_rad_minchi2_final = FloatField()
    v_rad_pix_var = FloatField()
    v_rad_pixoff_disc_final = FloatField()
    v_rad_pixoff_final = FloatField()
    v_rad_chi2_residuals = FloatField()

    #> APOGEE DR17 DRP Metadata
    drp_snr = FloatField()
    drp_starflag = BitField(default=0)
    drp_vhelio = FloatField()
    drp_vrel = FloatField()
    drp_vrelerr = FloatField()
    dr17_teff = FloatField()
    dr17_logg = FloatField()
    dr17_x_h = FloatField()
    dr17_vsini = FloatField()

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )

class ARJLTHVisitSpectrum(ARJLVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLFluxAccessor)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor)

class ARJLDDVisitSpectrum(ARJLVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLFluxAccessor)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor)



def shift(array, pixels, fill_value=0):
    pixels = np.asarray(pixels).astype(int)

    if pixels.ndim == 0:
        pixels = int(pixels)
        pad = fill_value * np.ones(abs(pixels))
        if pixels >= 0:
            return np.hstack([array[pixels:], pad])
        else:
            return np.hstack([pad, array[:pixels]])

    N, P = array.shape
    col_idx = np.arange(P)
    src_idx = col_idx[np.newaxis, :] + pixels[:, np.newaxis]  # (N, P)

    valid = (src_idx >= 0) & (src_idx < P)
    row_idx = np.arange(N)[:, np.newaxis] * np.ones((1, P), dtype=int)

    out = np.where(valid, array[row_idx, np.clip(src_idx, 0, P - 1)], fill_value)
    return out


transform_to_rest = lambda a, _, instance: shift(a, instance.v_rad_pixoff_final)

class ARJLTHRestFrameVisitSpectrum(ARJLTHVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLFluxAccessor, transform=transform_to_rest)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor, transform=transform_to_rest)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor, transform=transform_to_rest)


class ARJLDDRestFrameVisitSpectrum(ARJLDDVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLFluxAccessor, transform=transform_to_rest)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor, transform=transform_to_rest)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor, transform=transform_to_rest)


class ARJLTHStarLinesVisitSpectrum(ARJLTHVisitSpectrum):
    # ivar uses ARJLInverseVarianceAccessor (same as ARJLTHVisitSpectrum), not
    # ARJLStarLinesInverseVarianceAccessor: the posterior uncertainty of x_starLines_v0
    # badly underestimates the true errors, so we borrow the TH error model as a stand-in.
    flux = PixelArray(accessor_class=ARJLStarLinesFluxAccessor)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor)


class ARJLDDStarLinesVisitSpectrum(ARJLDDVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLStarLinesFluxAccessor)
    ivar = PixelArray(accessor_class=ARJLStarLinesInverseVarianceAccessor)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor)


class ARJLTHStarLinesRestFrameVisitSpectrum(ARJLTHStarLinesVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLStarLinesFluxAccessor, transform=transform_to_rest)
    ivar = PixelArray(accessor_class=ARJLInverseVarianceAccessor, transform=transform_to_rest)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor, transform=transform_to_rest)


class ARJLDDStarLinesRestFrameVisitSpectrum(ARJLDDStarLinesVisitSpectrum):
    flux = PixelArray(accessor_class=ARJLStarLinesFluxAccessor, transform=transform_to_rest)
    ivar = PixelArray(accessor_class=ARJLStarLinesInverseVarianceAccessor, transform=transform_to_rest)
    pixel_flags = PixelArray(accessor_class=ARJLPixelFlagsAccessor, transform=transform_to_rest)


def _get_coadd_array(dir, basename, key, index):
    with h5.File(f"{dir}/{basename}", "r") as fp:
        return fp[key][index]


class ARJLCoaddedPixelArrayAccessor(BasePixelArrayAccessor):

    """
    A class to access ARJL coadd pixel arrays (flux, ivar, pixel_flags).

    Unlike the visit accessors above, this data doesn't come from the DRP's apMADGICS
    outputs: it's computed by Astra itself (see `astra.migrations.arjl.build_arjl_coadds`)
    and written to a small per-batch HDF5 file that we own, with one row per (source, telescope).

    The filename comes from `instance.coadd_basename`, not from these accessor kwargs: TH/DD
    and StarLines coadds are different tables (see the four leaf classes below), each writing
    to its own file in the same `component_dir` so they don't collide with one another.
    """

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                value = _get_coadd_array(instance.component_dir, instance.coadd_basename, self.column_name, instance.row_index)
                if self.transform is not None:
                    value = self.transform(value, None, instance)
                instance.__pixel_data__.setdefault(self.name, value)
            finally:
                return instance.__pixel_data__[self.name]

        return self.field


class ARJLCoaddedSpectrum(BaseModel, SpectrumMixin):

    """
    A per-source, per-telescope pixel-weighted coadd of ApogeeReduction.jl visit spectra.

    DRAFT. Open questions before this is real:
      - What defines a 'good' visit for `n_good_visits` / the RV statistics? (e.g. some cut
        on `v_rad_flags`, `drp_starflag`, or `v_rad_chi2_residuals`)
      - ARJL visit flux is already continuum-normalized, but that normalization may not be
        consistent between exposures of the same star. Revisit whether visits need to be
        rescaled before stacking (see `scale_by_pseudo_continuum` on `pixel_weighted_spectrum`).
      - Do TH/DD and StarLines-only coadds deserve four near-identical tables (as with the
        visit spectra), or should this collapse into one table with a `kind` column?
      - Is per-telescope grouping the right key, or should apo25m/lco25m visits of the same
        star ever be combined?
    """

    pk = AutoField()

    # Not a DB column: which per-batch HDF5 file (in `component_dir`) this leaf model's
    # pixel arrays live in. Every subclass below must set its own, distinct value so that
    # TH/DD and StarLines coadds don't collide when they share the same `component_dir`.
    coadd_basename = "arjlStarCoadd.h5"

    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        column_name="spectrum_pk"
    )
    source = ForeignKeyField(
        Source,
        null=True,
        index=True,
        column_name="source_pk",
        backref="arjl_coadded_spectra",
    )

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    sdss_id = BigIntegerField(index=True, null=True)
    catalogid = BigIntegerField(index=True, null=True)

    #> Storage (see `ARJLCoaddedPixelArrayAccessor`)
    component_dir = TextField(null=False)
    row_index = IntegerField(index=True, null=False)
    v_arjl = TextField(null=True)

    #> Data Product Keywords
    release = TextField(index=True)
    telescope = TextField(index=True)

    #> Observing Span
    min_mjd = IntegerField(null=True)
    max_mjd = IntegerField(null=True)

    #> Number and Quality of Visits
    n_visits = IntegerField(null=True)
    n_good_visits = IntegerField(null=True)

    #> Summary Statistics
    snr = FloatField(null=True)
    mean_fiber = FloatField(null=True)
    std_fiber = FloatField(null=True)
    v_rad_flags = BitField(default=0)  # bitwise-OR of visit v_rad_flags
    drp_starflag = BitField(default=0)  # bitwise-OR of visit drp_starflag

    #> Radial Velocity
    # Inverse-variance weighted mean of visit v_rad, using v_rad_pix_var (converted from
    # pixels to km/s) as the per-visit weight -- see weighted_average() in migrations/arjl.py.
    v_rad = FloatField(null=True)
    e_v_rad = FloatField(null=True)
    std_v_rad = FloatField(null=True)

    #> APOGEE DR17 DRP Metadata
    # Carried over from one of the coadded visits (it's a per-star crossmatch value, not
    # per-visit, so it's the same across all visits of a star). ASPCAP's
    # get_initial_arjl_guesses() reads these to seed its initial guess -- without them here,
    # running aspcap on this model raises AttributeError before any FERRE work happens.
    dr17_teff = FloatField(null=True)
    dr17_logg = FloatField(null=True)
    dr17_x_h = FloatField(null=True)
    dr17_vsini = FloatField(null=True)

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )
    flux = PixelArray(accessor_class=ARJLCoaddedPixelArrayAccessor)
    ivar = PixelArray(accessor_class=ARJLCoaddedPixelArrayAccessor)
    pixel_flags = PixelArray(accessor_class=ARJLCoaddedPixelArrayAccessor)

    class Meta:
        indexes = (
            (
                (
                    "sdss_id",
                    "release",
                    "v_arjl",
                    "telescope",
                ),
                True,
            ),
        )


class ARJLTHCoaddedSpectrum(ARJLCoaddedSpectrum):
    """Coadd of `ARJLTHRestFrameVisitSpectrum` visits."""
    coadd_basename = "arjlStarCoadd_th.h5"


class ARJLDDCoaddedSpectrum(ARJLCoaddedSpectrum):
    """Coadd of `ARJLDDRestFrameVisitSpectrum` visits."""
    coadd_basename = "arjlStarCoadd_dd.h5"


class ARJLTHStarLinesCoaddedSpectrum(ARJLCoaddedSpectrum):
    """Coadd of `ARJLTHStarLinesRestFrameVisitSpectrum` visits."""
    coadd_basename = "arjlStarCoadd_th_starlines.h5"


class ARJLDDStarLinesCoaddedSpectrum(ARJLCoaddedSpectrum):
    """Coadd of `ARJLDDStarLinesRestFrameVisitSpectrum` visits."""
    coadd_basename = "arjlStarCoadd_dd_starlines.h5"
