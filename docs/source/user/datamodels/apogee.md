# APOGEE Spectrum Models

**Module:** `astra.models.apogee`

APOGEE spectra are infrared spectra (H-band, ~1.5--1.7 micron) with 8575 pixels on a log-lambda wavelength grid. There are three APOGEE spectrum models in Astra, each corresponding to a different stage of processing.

## ApogeeVisitSpectrum

An individual visit spectrum from the APOGEE data reduction pipeline, stored in an `apVisit` file. A "visit" is a single observation of a source (possibly combining multiple dithered exposures within the same night).

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `spectrum_pk` | ForeignKeyField | Unique spectrum identifier, links to the `Spectrum` table |
| `source` | ForeignKeyField | Foreign key to the `Source` this spectrum belongs to |
| `release` | TextField | Data release (e.g., `"sdss5"` or `"dr17"`) |
| `apred` | TextField | APOGEE reduction pipeline version |
| `telescope` | TextField | Telescope used (e.g., `"apo25m"`, `"lco25m"`, `"apo1m"`) |
| `plate` | TextField | Plate identifier |
| `field` | TextField | Field identifier |
| `fiber` | IntegerField | Fiber number |
| `mjd` | IntegerField | Modified Julian Date of observation |
| `obj` | TextField | Object name (computed per-spectrum from input RA/Dec) |
| `snr` | FloatField | Signal-to-noise ratio |
| `wavelength` | PixelArray | Wavelength array (vacuum, observed frame) |
| `flux` | PixelArray | Flux array |
| `ivar` | PixelArray | Inverse variance array |
| `pixel_flags` | PixelArray | Per-pixel quality flags |

**Radial velocity fields:**

| Field | Type | Description |
|-------|------|-------------|
| `v_rad` | FloatField | Absolute radial velocity (km/s) |
| `v_rel` | FloatField | Relative radial velocity (km/s) |
| `e_v_rel` | FloatField | Uncertainty on relative radial velocity (km/s) |
| `bc` | FloatField | Barycentric correction (km/s) |
| `doppler_teff` | FloatField | Effective temperature from Doppler RV code |
| `doppler_logg` | FloatField | Surface gravity from Doppler RV code |
| `doppler_fe_h` | FloatField | Metallicity from Doppler RV code |

**Quality flags:**

| Field | Type | Description |
|-------|------|-------------|
| `visit_flags` | BitField | Bit field encoding quality warnings from apVisit product |
| `star_flags` | BitField | Bit field encoding quality warnings from apStar product |
| `flag_bad_pixels` | bit 0 | Spectrum has many bad pixels |
| `flag_bright_neighbor` | bit 2 | Star has a neighbor more than 10x brighter |
| `flag_very_bright_neighbor` | bit 3 | Star has a neighbor more than 100x brighter |
| `flag_low_snr` | bit 4 | Low S/N spectrum |
| `flag_rv_failure` | bit 22 | RV failure |

### Methods and Properties

#### `path` (property)

Returns the file path to the `apVisit` FITS file for this spectrum. The path template depends on `release` and `telescope`.

```python
visit = ApogeeVisitSpectrum.get_by_id(1)
print(visit.path)
# e.g., "$SAS_BASE_DIR/ipl-4/spectro/apogee/redux/1.3/visit/apo25m/..."
```

#### `get_path_template(release, telescope)` (classmethod)

Returns the file path template string for a given release and telescope combination.

#### `flag_bad` (hybrid property)

Returns `True` if any of the critical quality flags are set (`flag_bad_pixels`, `flag_very_bright_neighbor`, `flag_bad_rv_combination`, `flag_rv_failure`). Can be used in database queries.

#### `flag_warn` (hybrid property)

Returns `True` if any spectrum flag is set. Can be used in database queries.


## ApogeeVisitSpectrumInApStar

A visit spectrum as stored within an `apStar` file. Unlike the raw `apVisit`, these spectra have been resampled onto a common log-lambda wavelength grid and shifted to the source rest frame.

### Key Fields

Fields are similar to `ApogeeVisitSpectrum`, with these additions:

| Field | Type | Description |
|-------|------|-------------|
| `drp_spectrum_pk` | ForeignKeyField | Links back to the original `ApogeeVisitSpectrum` |
| `apstar` | TextField | apStar version (default: `"stars"`) |
| `healpix` | IntegerField | HEALPix index (may differ from Source due to APOGEE DRP computation) |

**Spectral data arrays:**

| Field | Type | Description |
|-------|------|-------------|
| `wavelength` | PixelArray | Log-lambda wavelength grid (8575 pixels, crval=4.179, cdelt=6e-6) |
| `flux` | PixelArray | Flux on the common grid |
| `ivar` | PixelArray | Inverse variance on the common grid |
| `pixel_flags` | PixelArray | Per-pixel quality flags |

### Methods and Properties

#### `path` (property)

Returns the file path to the `apStar` FITS file. Path templates differ between `"sdss5"` and `"dr17"` releases.


## ApogeeCoaddedSpectrumInApStar

A co-added (stacked) APOGEE spectrum from an `apStar` file, created by combining all good visit spectra for a source. This represents the highest signal-to-noise APOGEE spectrum available for a given source.

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `spectrum_pk` | ForeignKeyField | Unique spectrum identifier |
| `source` | ForeignKeyField | Foreign key to the `Source` |
| `star_pk` | BigIntegerField | APOGEE DRP star primary key |
| `release` | TextField | Data release |
| `apred` | TextField | APOGEE reduction pipeline version |
| `obj` | TextField | Object name |
| `telescope` | TextField | Telescope used |

**Observation summary:**

| Field | Type | Description |
|-------|------|-------------|
| `n_entries` | IntegerField | Number of entries in apStar file |
| `n_visits` | IntegerField | Number of visits |
| `n_good_visits` | IntegerField | Number of good visits |
| `n_good_rvs` | IntegerField | Number of good RV measurements |
| `min_mjd` | IntegerField | Minimum MJD of contributing visits |
| `max_mjd` | IntegerField | Maximum MJD of contributing visits |

**Summary statistics and radial velocity:**

| Field | Type | Description |
|-------|------|-------------|
| `snr` | FloatField | Signal-to-noise ratio of co-added spectrum |
| `mean_fiber` | FloatField | S/N-weighted mean fiber number |
| `std_fiber` | FloatField | Standard deviation of fiber numbers |
| `v_rad` | FloatField | Mean radial velocity (km/s) |
| `e_v_rad` | FloatField | Uncertainty on mean radial velocity (km/s) |
| `std_v_rad` | FloatField | Scatter in radial velocity across visits (km/s) |
| `doppler_teff` | FloatField | Effective temperature from Doppler |
| `doppler_logg` | FloatField | Surface gravity from Doppler |
| `doppler_fe_h` | FloatField | Metallicity from Doppler |

**Spectral data:**

| Field | Type | Description |
|-------|------|-------------|
| `wavelength` | PixelArray | Log-lambda wavelength grid (8575 pixels, crval=4.179, cdelt=6e-6) |
| `flux` | PixelArray | Co-added flux |
| `ivar` | PixelArray | Co-added inverse variance |
| `pixel_flags` | PixelArray | Per-pixel quality flags |

### Methods and Properties

#### `path` (property)

Returns the file path to the `apStar` FITS file containing this co-added spectrum.

```python
coadd = ApogeeCoaddedSpectrumInApStar.get_by_id(1)
print(coadd.path)
```
