# Data models

This page describes data models that are frequently given to, or produced by, Astra.

## APOGEE pipeline products

### `apVisit`

An `apVisit` file contains a 1D infrared APOGEE spectrum of a single "visit" of a source, where a visit might include multiple dithered exposures. The wavelengths are in vacuum and in the observed frame: no correction is made based on the relative velocity of the source. 

These files are produced by the [APOGEE data reduction pipeline](https://github.com/sdss/apogee_drp). 

These file types exist in SDSS-IV and in SDSS-V. The keywords needed to identify an `apVisit` file are different between data release 16 (SDSS-IV) and SDSS-V:
```python
>> from sdss_access import SDSSPath

>> SDSSPath("dr16").lookup_keys("apVisit")
   ['field', 'telescope', 'prefix', 'mjd', 'plate', 'fiber', 'apred']

>> SDSSPath("sdss5").lookup_keys("apVisit")
   ['field', 'telescope', 'mjd', 'plate', 'fiber', 'apred']
```

### `apStar`

An `apStar` file includes infrared APOGEE 1D spectra of a source (usually a star). If a source was observed many times (i.e., has many _visits_) and those spectra were considered useful, then those spectra will be included in the `apStar` file. However, unlike the `apVisit` files, the spectra in the `apStar` files are re-sampled onto a common wavelength array (vacuum) and are redshifted to the source rest frame. 

The number of useful visit spectra will not always match the number of spectra in the `apStar` file. If there is only one useful visit, then there will be only one spectrum in the `apStar` file. If there are multiple useful visits then these will be stacked together to create a high signal-to-noise ratio spectrum. As of September 2022 there are two methods used to create a stacked spectrum: a pixel-weighting method, and a global-weighting method. For this reason, if there are 2 useful visit spectra then the `apStar` file will contain four spectra: a pixel-weighted stacked spectrum, a global-weighted stacked spectra, and the two visit spectra. The definition of whether a spectrum was considered 'useful' or not depends on a minimum signal-to-noise requirement (>3), it requires that the spectrum was not flagged by the APOGEE data reduction pipeline as being 'bad', and that the radial velocity determination succeeded, among a few other things.

These files are produced by the [APOGEE data reduction pipeline](https://github.com/sdss/apogee_drp). 

These file types exist in SDSS-IV and in SDSS-V. The keywords needed to identify an `apStar` file are different between data release 16 (SDSS-IV) and SDSS-V:
```python
>> from sdss_access import SDSSPath

>> SDSSPath("dr16").lookup_keys("apStar")
   ['telescope', 'prefix', 'field', 'obj', 'apstar', 'apred']

>> SDSSPath("sdss5").lookup_keys("apStar")
   ['telescope', 'obj', 'healpix', 'apstar', 'apred']
```

## BOSS pipeline products

### `specFull`

A `specFull` file includes a 1D optical BOSS spectrum of a source. This spectrum might often be a co-add of multiple exposures of the source during the same night. The wavelengths are not resampled onto a common array, but are given in vacuum and are shifted to the rest frame of the Solar system barycentre. 

These files are produced by the [BOSS data reduction pipeline](https://github.com/sdss/idlspec2d).

These files are the closest optical analogue of `apVisit` files for APOGEE spectra. There are no data products produced by the BOSS data reduction pipeline that stack spectra of the same source taken over multiple nights (e.g., if `specFull` is the optical analogue of `apVisit`, then there is no equivalent analogue of `apStar`).

These files only exist in SDSS-V. The keywords needed to identify a `specFull` file are:
```python
>> from sdss_access import SDSSPath

>> SDSSPath("sdss5").lookup_keys("specFull")
   ['isplate', 'mjd', 'catalogid', 'fieldid', 'run2d']
```

## Astra products

### Observed data products

Astra produces some data products that only contain observations (`mwmVisit` and `mwmStar`) in order to create high signal-to-noise stacked BOSS spectra (which are not produced by the BOSS data reduction pipeline), and to streamline differences in the data products produced by the BOSS and APOGEE data reduction pipelines.

#### `mwmVisit`

The `mwmVisit` file contains **all spectra** for a Milky Way Mapper (MWM) source, from all telescope/instrument combinations. It has 5 header data units (HDUs):

0. A primary HDU containing information about the source (e.g., photometry, astrometry).
1. All BOSS spectra from Apache Point Observatory
2. All BOSS spectra from Las Campanas Observatory
3. All APOGEE spectra from Apache Point Observatory
4. All APOGEE spectra from Las Campanas Observatory

The `mwmStar` files are similar in form, but they contain only a single stacked spectrum per HDU instead of individual visits. All `mwmVisit` and `mwmStar` spectra are redshifted to the source rest frame, and resampled onto a common wavelength array.

Creating a `mwmVisit` data product requires `apVisit` products and `specFull` products as inputs. However, the `mwmVisit` files differ slightly to `apVisit` and `specFull` files in a few important ways:
- The wavelengths of all `mwmVisit` spectra are in the source rest frame. The `apStar` spectra were all in the source rest frame, but `apVisit` spectra are in the observed frame, and `specFull` are in the Solar system barycentric rest frame.
- The `mwmVisit` spectra are re-sampled onto a common wavelength array. The `apStar` spectra were resampled onto a common wavelength array, and the `mwmVisit` files use that same sampling. However, the `apVisit` spectra are not resampled, and neither are the `specFull` files.
- If an `apVisit` spectrum was deemed 'unreliable' then it would not be used to create the stacked spectra in the `apStar` file, and the `apVisit` spectrum would not appear as a row in the `apStar` file. In this case the spectrum **does exist** in the `mwmVisit` file, and an accompanying boolean column called `IN_STACK` describes whether the spectrum was used to create a stacked spectrum in the `mwmStar` file.

These differences mean that `mwmVisit` files include **all the data** that SDSS-V collects for a source, resampled onto a wavelength array that is ready for scientific analysis. And even if some spectra were not considered reliable (for whatever reason), those spectra are included for the user to examine.

The `mwmVisit` files can be given to nearly every component in Astra. `mwmVisit` files only exist in SDSS-V. The keywords required to identify a `mwmVisit` file are:

- `catalogid`: the SDSS-V catalog identifier
- `apred`: the APOGEE data reduction pipeline version used to create the input `apVisit` files
- `run2d`: the BOSS data reduction pipeline used to create the input `specFull` files
- `astra_version`: the version of Astra that created the file

```{Note}
**Spectroscopic binaries**

In future the `mwmVisit` and `mwmStar` products will have an optional keyword `component` to describe a stellar component that has been disentangled from the existing spectra. For example, in the case of a spectroscopic binary, the individual spectra of the two stars can often be disentangled by using multiple visits. In these cases there will be additional data products produced by a component of Astra. If the `catalogid` is 123456789 then the `mwmVisit` filenames will be:

1. `mwmVisit-...-123456789.fits`: the original spectra without any disentangling (e.g., `component=None` or `component=''`)
2. `mwmVisit-...-123456789A.fits`: the disentangled spectra of the primary component, A (e.g., `component='A'`)
3. `mwmVisit-...-123456789B.fits`: the disentangled spectra of the secondary component, B (e.g., `component='B'`)

These data products can be processed by Astra components as if they were normal observations, allowing us to see what happens when we feed in known spectroscopic binary spectra (example 1), and obtain stellar parameters of the individual (disentangled) components (examples 2 and 3).

The original spectra without disentangling will always be available, because disentangling is a model-dependent process. This note also applies to `mwmStar` spectra.
```

#### `mwmStar`

A `mwmStar` file contains **all stacked spectra** for a MWM source. This includes a stacked optical BOSS spectrum, and a stacked infrared spectrum from the APOGEE instrument at Apache Point Observatory, and a stacked infrared spectrum from the APOGEE instrument on the du Pont telescope at Las Campanas Observatory. Currently we do not stack spectra from different APOGEE instruments because of their different line spread function profiles. Stacked spectra from each telescope/instrument combination are stored in their own HDU (see `mwmVisit` HDU definition).

The `mwmVisit` and `mwmStar` files are created simultaneously so that they will always remain synchronised. Any spectra in the `mwmVisit` file that has `IN_STACK` marked as `True` contributed to the stacked spectrum in the `mwmStar` file. The `mwmVisit` and `mwmStar` files have the same (resampled) wavelength array, which is in vacuum and in the source rest frame. The `mwmStar` spectra are ready for scientific analysis, **and represent our current "best spectrum" for a source**. 

The `mwmStar` files can be given to nearly every component in Astra. `mwmStar` files exist only in SDSS-V. The keywords required to identify a `mwmStar` file are the same required for a `mwmVisit` file:

- `catalogid`: the SDSS-V catalog identifier
- `apred`: the APOGEE data reduction pipeline version used to create the input `apVisit` files
- `run2d`: the BOSS data reduction pipeline used to create the input `specFull` files
- `astra_version`: the version of Astra that created the file

### Component-level products

Some analysis components in Astra only produce estimates of stellar properties, without any accompanying model spectra. Other components produce both model spectra and stellar properties. The `astraVisit` and `astraStar` files store best-fitting model spectra (from a single component) for a given source. These contain the model spectra that are companions to the `mwmVisit` and `mwmStar` observed data products.

#### `astraVisit`

Documentation TBD. Very similar to `mwmVisit`, but contains best-fitting model spectrum from an analysis pipeline.

#### `astraStar`

Documentation TBD. Very similar to `mwmStar`, but contains best-fitting model spectrum from an analysis pipeline.

### Summary products

#### `astraAllStar`

Documentation TBD. Contains stellar parameter estimates from one or many pipelines. There could/should be two types of files:
- per-pipeline: results for all stars (from `mwmStar` high S/N spectrum) from a given pipeline
- per-carton: results for all stars (from `mwmStar` high S/N spectrum) in a carton, split between pipelines in a way chosen by carton owners, with separate HDUs for each pipeline used, and no stars with results from multiple pipelines (`astraAllStarBest`?)


#### `astraAllVisit`

Documentation TBD. Contains stellar parameter estimates from one or many pipelines. 