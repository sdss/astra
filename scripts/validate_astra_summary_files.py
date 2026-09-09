#!/usr/bin/env python3
"""Analyze FITS files and produce HTML QA reports."""

import argparse
import html
import os
import re
import traceback
import warnings
from collections import Counter, defaultdict

import numpy as np
from astropy.io import fits

from astra import __version__
from astra.utils import expand_path

warnings.filterwarnings("ignore")

SUMMARY_DIR = expand_path(f"$MWM_ASTRA/{__version__}/summary")
VALIDATION_DIR = expand_path(f"$MWM_ASTRA/{__version__}/validation")

RANGE_CHECKS = {
    "ra": (0, 360, "degrees"),
    "dec": (-90, 90, "degrees"),
    "glon": (0, 360, "degrees"),
    "glat": (-90, 90, "degrees"),
    "teff": (100, 100000, "K"),
    "logg": (-2, 10, "dex"),
    "fe_h": (-10, 2, "dex"),
    "m_h": (-10, 2, "dex"),
    "alpha_m": (-3, 3, "dex"),
    "v_rad": (-1500, 1500, "km/s"),
    "vrad": (-1500, 1500, "km/s"),
    "v_rel": (-1500, 1500, "km/s"),
    "radial_velocity": (-1500, 1500, "km/s"),
    "snr": (0, 100000, ""),
    "meanfib": (0, 1000, ""),
    "plx": (-500, 500, "mas"),
    "parallax": (-500, 500, "mas"),
    "pmra": (-10000, 10000, "mas/yr"),
    "pmdec": (-10000, 10000, "mas/yr"),
}

# Sentinels split by how safely they can be distinguished from a real measurement.
# UNAMBIGUOUS values sit far outside every physical range we care about, so they can be
# reported for any numeric column. AMBIGUOUS values overlap real astrophysics: -1 is an
# ordinary [Fe/H], a plausible v_rad, and a normal magnitude, and -9.999/-99.999/99.999
# collide with abundances and velocities too. Reporting those on a float measurement
# column produces false positives, so they are only used where a fill value is genuinely
# distinguishable (integer/index columns and columns with no known physical range).
UNAMBIGUOUS_SENTINELS = {-9999.0, -999.0, 9999.0}
AMBIGUOUS_SENTINELS = {-99.999, -9.999, 99.999, -1}
SENTINELS = UNAMBIGUOUS_SENTINELS | AMBIGUOUS_SENTINELS

# Suffixes that mark a derived/bookkeeping quantity rather than the physical measurement
# whose name it borrows. "fe_h_flags" is a bitmask, "fe_h_rchi2" a goodness-of-fit
# statistic, "zgr_fe_h_confidence" a 0-1 score; none of them are dex and none should
# inherit the [Fe/H] range.
NON_PHYSICAL_SUFFIXES = (
    "_flags", "_flag", "_rchi2", "_chi2", "_chisq", "_confidence", "_id", "_pk",
    "_index", "_order", "_mask", "_bitmask", "_quality", "_status", "_count",
)

# Suffixes/prefixes that mark a column as an uncertainty, scatter or spread. These are
# non-negative by construction and must NOT inherit the range of the value they describe
# (e_teff is an uncertainty in K, not a temperature, so the 100 K floor is meaningless).
ERROR_SUFFIXES = ("_err", "_error", "_unc", "_uncertainty", "_sigma", "_scatter")
ERROR_TOKEN_PREFIXES = ("e_", "err_", "sigma_", "std_", "stddev_")

# Columns that are constant by construction, so zero variance is expected, not a defect.
CONSTANT_OK_COLUMNS = {"v_astra", "release", "created", "tag", "reduction", "healpix_nside"}

# HDU 1 is the BOSS table in your files, HDU 2 is the APOGEE-style combined table.
BOSS_APOGEE_MISSING_OK = {"n_apogee_visits", "apogee_min_mjd", "apogee_max_mjd"}
APOGEE_MISSING_OK = {"n_boss_visits", "boss_min_mjd", "boss_max_mjd"}

SEVERITY_RANK = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}


def normalize_value(value):
    # Convert FITS values to plain text for comparisons and display.
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value).strip()


def normalize_number(value):
    # Keep numeric stats compact and readable.
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
    return value


def issue_signature(colname, category, release_label=None, extra=None):
    # Build a stable key for duplicate removal.
    """Stable key for deduping; ignore counts/percentages and release labels."""
    key = (normalize_value(colname).lower(), normalize_value(category).lower())
    if extra is not None:
        key += (normalize_value(extra).lower(),)
    return key


def issue(hdu_index, hdu_name, colname, message, severity="INFO", *, release_label=None, category=None, pct=None, extra=None):
    # Store the verbose message plus a normalized signature.
    return {
        "hdu_index": hdu_index,
        "hdu_name": hdu_name or "",
        "colname": colname or "",
        "release_label": release_label,
        "severity": severity,
        "message": message,
        "category": category or message,
        "signature": issue_signature(colname, category or message, release_label=release_label, extra=extra),
        "pct": pct,
    }


def sort_key(item):
    # Sort by severity first, then by impact.
    pct = item.get("pct")
    if isinstance(pct, (int, float, np.integer, np.floating)):
        pct_sort = -float(pct)
    else:
        pct_sort = 1e9
    severity_sort = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(item.get("severity", "INFO"), 99)
    return (severity_sort, pct_sort, item.get("order", 1e9))


def summary_sort_key(item):
    # Top-level summaries only show critical and warning.
    pct = item.get("pct")
    if isinstance(pct, (int, float, np.integer, np.floating)):
        pct_sort = -float(pct)
    else:
        pct_sort = 1e9
    severity_sort = {"CRITICAL": 0, "WARNING": 1}.get(item.get("severity", "WARNING"), 99)
    return (severity_sort, pct_sort, item.get("order", 1e9))


def is_error_column(colname):
    # Decide whether a column holds an uncertainty rather than a measurement.
    # Match on underscore-delimited tokens, never on a bare substring: "e_" appears
    # inside fe_h, ce_h, coarse_*, ferre_*, apogee_*, source_pk and many more, and a
    # substring test silently classifies all of them as error columns.
    cl = colname.lower().strip()
    if cl.endswith(ERROR_SUFFIXES):
        return True
    for prefix in ERROR_TOKEN_PREFIXES:
        # Either the whole name starts with the token (e_teff) or it appears as an
        # interior token after a pipeline prefix (zgr_e_teff, doppler_e_teff, gaia_e_v_rad).
        if cl.startswith(prefix) or ("_" + prefix) in cl:
            return True
    return False


def is_non_physical_column(colname):
    # Bitmasks, fit statistics and identifiers borrow the name of a physical quantity but
    # carry none of its units, so they must not inherit its expected range.
    cl = colname.lower().strip()
    return cl.endswith(NON_PHYSICAL_SUFFIXES)


def get_range_check(colname):
    # Map column names to simple expected ranges.
    cl = colname.lower().strip()

    # Order matters. Error and non-physical columns are resolved first, because the
    # fuzzy prefix/suffix match below would otherwise hand e_teff the temperature range
    # and fe_h_flags the [Fe/H] range.
    if is_non_physical_column(cl):
        return None
    if is_error_column(cl):
        return (0, None, "error column")

    if cl in RANGE_CHECKS:
        return RANGE_CHECKS[cl]
    for key, val in RANGE_CHECKS.items():
        if cl == key or cl.startswith(key + "_") or cl.endswith("_" + key):
            return val
    return None


def format_number(value):
    # Format stats without forcing everything to a float string.
    if value is None or value == "":
        return ""
    value = normalize_number(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        if abs(value) >= 1e6 or (0 < abs(value) < 1e-3):
            return f"{value:.4g}"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return html.escape(str(value))


def is_string_array(data):
    # String arrays need separate missing-value handling.
    return data.dtype.kind in ("U", "S", "O")


def looks_like_missing_string(value):
    # Treat blank strings as missing.
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip() == ""


def is_flag_name(colname):
    # Name-only test, used by the quality checks that must skip bitmask columns.
    name = normalize_value(colname).lower()
    return name.endswith("_flags") or name.endswith("_flag") or name.endswith("flags")


def is_flag_column(colname, col_format, data):
    # Flags are integer bitmasks. Accept every integer FITS width, not just K:
    # w1uflags/w2aflags are K but have no underscore, and the *_mag_flag columns are J.
    name = normalize_value(colname).lower()
    fmt = normalize_value(col_format).upper()
    if not is_flag_name(name):
        return False
    # Strip any repeat count ("73B") before looking at the type code.
    code = fmt.lstrip("0123456789")
    return code.startswith(("K", "J", "I", "B"))


def compute_flag_statistics(data, colname, col_format):
    # Count how often each bit is set for a flag bitmask column.
    if not is_flag_column(colname, col_format, data):
        return None

    arr = np.asarray(data)
    if arr.size == 0:
        return None

    if arr.dtype.kind not in ("i", "u"):
        try:
            arr = arr.astype(np.int64)
        except Exception:
            return None

    valid = arr.astype(np.int64, copy=False)
    valid = valid[~np.isin(valid, list(SENTINELS))]
    if valid.size == 0:
        return None

    total = valid.size
    zero_count = int(np.sum(valid == 0))
    zero_pct = 100.0 * zero_count / total if total else 0.0

    max_value = int(np.max(valid))
    nbits = max(1, max_value.bit_length())
    bits = []
    for bit in range(nbits):
        mask = (valid & (1 << bit)) != 0
        count = int(np.sum(mask))
        bits.append({
            "bit": bit,
            "count": count,
            "pct": 100.0 * count / total if total else 0.0,
        })

    return {
        "name": normalize_value(colname),
        "format": normalize_value(col_format),
        "total_rows": total,
        "max_value": max_value,
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "bits": bits,
    }


def top_level_issues(results):
    # Keep info in the HDU tables, but hide it from the top summary cards and issue list.
    return [issue for issue in results.get("all_issues", []) if issue.get("severity") in {"CRITICAL", "WARNING"}]

def analyze_column(data, colname, expected_max_mjd=None, *, hdu_index=None, total_hdus=None, release_label=None, allow_optional_missing=False, boss_downgrade=False):
    # Run the per-column checks and collect stats.
    # This function keeps the verbose HTML message separate from the normalized signature.
    issues = []
    stats = {"dtype": str(data.dtype), "dtype_kind": data.dtype.kind, "type": str(data.dtype)}
    total = len(data)
    stats["total"] = total

    if total == 0:
        return issues, stats

    # A zero-length column usually means the table was filtered down to nothing.

    # Strings use blank values instead of NaN, so handle them separately.
    # This catches columns that look populated in metadata but are really empty.
    if is_string_array(data):
        empty = 0
        try:
            if data.dtype.kind == "S":
                empty = int(np.sum(data == b""))  # fixed-width byte strings
            else:
                empty = int(np.sum([looks_like_missing_string(v) for v in data]))
        except Exception:
            empty = 0
        stats["empty"] = empty
        if empty == total:
            if not allow_optional_missing:
                issues.append(issue(hdu_index, "", colname, "Column is entirely empty strings", "WARNING", release_label=release_label, category="empty_strings", pct=100.0))
            return issues, stats
        if empty > total * 0.9 and not allow_optional_missing:
            pct = 100.0 * empty / total
            issues.append(issue(hdu_index, "", colname, f"{empty}/{total} ({pct:.1f}%) empty strings", "INFO", release_label=release_label, category="empty_strings", pct=pct))
        return issues, stats

    arr = data
    # Flatten multi-dimensional cells so the checks see all values the same way.
    # That avoids treating vector-valued cells as a special case in every check.
    if len(arr.shape) > 1:
        arr = arr.flatten()
        stats["shape"] = str(data.shape)

    total = len(arr)
    stats["total"] = total
    if total == 0:
        return issues, stats

    # A zero-length column usually means the table was filtered down to nothing.

    finite = arr
    nan_count = 0
    inf_count = 0

    if arr.dtype.kind == "f":
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        stats["nan_count"] = nan_count
        stats["inf_count"] = inf_count

        if inf_count > 0:
            pct = 100.0 * inf_count / total
            issues.append(issue(hdu_index, "", colname, f"{inf_count} Inf values ({pct:.1f}%)", "CRITICAL", release_label=release_label, category="inf_values", pct=pct))

        if nan_count == total:
            sev = "WARNING" if boss_downgrade else "CRITICAL"
            issues.append(issue(hdu_index, "", colname, "Column is entirely NaN", sev, release_label=release_label, category="all_nan", pct=100.0))
            return issues, stats

        if nan_count > total * 0.5:
            pct = 100.0 * nan_count / total
            issues.append(issue(hdu_index, "", colname, f"{nan_count}/{total} ({pct:.1f}%) NaN values", "WARNING", release_label=release_label, category="nan_fraction", pct=pct))
        elif nan_count > total * 0.01:
            pct = 100.0 * nan_count / total
            issues.append(issue(hdu_index, "", colname, f"{nan_count}/{total} ({pct:.1f}%) NaN values", "INFO", release_label=release_label, category="nan_fraction", pct=pct))

        finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        return issues, stats

    try:
        stats["min"] = normalize_number(np.min(finite))
        stats["max"] = normalize_number(np.max(finite))
        stats["mean"] = normalize_number(np.mean(finite))
        stats["std"] = normalize_number(np.std(finite))
        stats["median"] = normalize_number(np.median(finite))
    except Exception:
        return issues, stats

    cl = colname.lower().strip()
    is_flag = is_flag_name(cl)
    range_check = get_range_check(colname)

    # A bitmask is not a measurement: every value is a legal bit pattern, so sentinel,
    # zero-variance, all-zero and range checks are meaningless on it. All-zero in
    # particular is the healthy state (no flags raised), not a defect.
    # The ambiguous sentinels are only trustworthy where a fill value cannot be confused
    # with a real measurement: integer columns, or columns with no known physical range.
    if is_flag:
        applicable_sentinels = set()
    elif arr.dtype.kind in ("i", "u") or range_check is None:
        applicable_sentinels = SENTINELS
    else:
        applicable_sentinels = UNAMBIGUOUS_SENTINELS

    for sentinel in applicable_sentinels:
        try:
            count = int(np.sum(np.isclose(finite, sentinel, atol=0.01)))
        except Exception:
            continue
        if count <= 0:
            continue
        pct = 100.0 * count / total
        if count == total:
            sev = "WARNING" if boss_downgrade else "CRITICAL"
            issues.append(issue(hdu_index, "", colname, f"All values are sentinel value {sentinel}", sev, release_label=release_label, category=f"sentinel:{sentinel}", pct=100.0, extra=sentinel))
        elif pct > 50:
            issues.append(issue(hdu_index, "", colname, f"{count}/{total} ({pct:.1f}%) values equal to sentinel {sentinel}", "WARNING", release_label=release_label, category=f"sentinel:{sentinel}", pct=pct, extra=sentinel))
        elif pct > 1:
            issues.append(issue(hdu_index, "", colname, f"{count}/{total} ({pct:.1f}%) values equal to sentinel {sentinel}", "INFO", release_label=release_label, category=f"sentinel:{sentinel}", pct=pct, extra=sentinel))

    # Let the expected max MJD vary per run.
    # This is useful because the file-level max can change every execution.
    if expected_max_mjd is not None and cl.endswith("max_mjd"):
        real_values = finite
        for sentinel in SENTINELS:
            real_values = real_values[~np.isclose(real_values, sentinel, atol=0.01)]
        if len(real_values) > 0:
            above = int(np.sum(real_values > expected_max_mjd))
            if above > 0:
                pct = 100.0 * above / total
                sev = "WARNING" if pct > 10 else "INFO"
                issues.append(issue(hdu_index, "", colname, f"{above} values ({pct:.1f}%) above expected max MJD {expected_max_mjd} (max={format_number(stats['max'])})", sev, release_label=release_label, category="above_expected_max_mjd", pct=pct, extra=expected_max_mjd))

    all_zero = stats.get("min", 1) == 0 and stats.get("max", 1) == 0 and len(finite) > 0
    constant_ok = is_flag or cl in CONSTANT_OK_COLUMNS

    # "All values are zero" and "Zero variance" describe the same fact for an all-zero
    # column, so report the more specific one only.
    if all_zero and not constant_ok:
        issues.append(issue(hdu_index, "", colname, "All values are zero", "WARNING", release_label=release_label, category="all_zero"))
    elif stats.get("std", 1) == 0 and len(finite) > 1 and not constant_ok:
        if not any(np.all(np.isclose(finite, sentinel, atol=0.01)) for sentinel in SENTINELS):
            issues.append(issue(hdu_index, "", colname, f"Zero variance - all {len(finite)} finite values are identical ({stats['min']})", "WARNING", release_label=release_label, category="zero_variance"))

    if range_check and not is_flag:
        lo, hi, unit = range_check
        if lo is not None:
            bad = finite < lo
            # Match the tolerance used everywhere else. The old atol=1 masked every real
            # value within 1.0 of a sentinel, e.g. any genuine logg in [-2, 0].
            for sentinel in (-9999, -999, -1):
                bad &= ~np.isclose(finite, sentinel, atol=0.01)
            count = int(np.sum(bad))
            if count > 0:
                pct = 100.0 * count / total
                sev = "WARNING" if pct > 10 else "INFO"
                issues.append(issue(hdu_index, "", colname, f"{count} values ({pct:.1f}%) below expected minimum {lo} {unit} (min={format_number(stats['min'])})", sev, release_label=release_label, category="below_expected_min", pct=pct, extra=lo))
        if hi is not None:
            count = int(np.sum(finite > hi))
            if count > 0:
                pct = 100.0 * count / total
                sev = "WARNING" if pct > 10 else "INFO"
                issues.append(issue(hdu_index, "", colname, f"{count} values ({pct:.1f}%) above expected maximum {hi} {unit} (max={format_number(stats['max'])})", sev, release_label=release_label, category="above_expected_max", pct=pct, extra=hi))

    # Token-based, so abundances (fe_h, ce_h) and pipeline prefixes (coarse_, ferre_,
    # apogee_, source_) are no longer mistaken for uncertainty columns.
    if is_error_column(cl) and not is_non_physical_column(cl) and arr.dtype.kind == "f":
        bad = finite < 0
        for sentinel in SENTINELS:
            bad &= ~np.isclose(finite, sentinel, atol=0.01)
        count = int(np.sum(bad))
        if count > 0:
            pct = 100.0 * count / total
            sev = "WARNING" if pct > 10 else "INFO"
            issues.append(issue(hdu_index, "", colname, f"{count} negative error values ({pct:.1f}%)", sev, release_label=release_label, category="negative_error", pct=pct))

    return issues, stats


def analyze_hdu_table(hdu, hdu_index, total_hdus, expected_max_mjd=None, release_label=None, release_mask=None):
    # Analyze one table HDU, optionally on a release subset.
    # The same helper is used for the full table and for each release slice.
    """Analyze one table HDU, optionally on a row subset."""
    hdu_name = hdu.name or ""
    data = hdu.data
    nrows = int(data.shape[0]) if data is not None else 0
    columns = []
    issues = []
    flag_statistics = []

    if release_mask is not None:
        nrows = int(np.sum(release_mask))

    for col in hdu.columns:
        col_name = col.name
        col_info = {
            "name": col_name,
            "format": col.format,
            "unit": getattr(col, "unit", "") or "",
            "dtype": str(getattr(col, "dtype", "")),
            "issues": [],
            "stats": {},
        }

        if data is None or nrows == 0:
            columns.append(col_info)
            continue

        try:
            col_data = data[col_name]
            if release_mask is not None:
                col_data = col_data[release_mask]

            allow_optional_missing = (total_hdus == 3 and hdu_index == 2 and col_name.lower() in APOGEE_MISSING_OK)
            boss_downgrade = (total_hdus == 3 and hdu_index == 1 and col_name.lower() in BOSS_APOGEE_MISSING_OK)

            col_issues, col_stats = analyze_column(
                col_data,
                col_name,
                expected_max_mjd=expected_max_mjd,
                hdu_index=hdu_index,
                total_hdus=total_hdus,
                release_label=release_label,
                allow_optional_missing=allow_optional_missing,
                boss_downgrade=boss_downgrade,
            )
            col_info["issues"] = col_issues
            col_info["stats"] = col_stats

            if release_label is None:
                flag_stat = compute_flag_statistics(col_data, col_name, col.format)
                if flag_stat is not None:
                    flag_statistics.append(flag_stat)

            for it in col_issues:
                issue_copy = dict(it)
                issue_copy["hdu_index"] = hdu_index
                issue_copy["hdu_name"] = hdu_name
                issue_copy["colname"] = col_name
                issue_copy["release_label"] = release_label
                issue_copy["message"] = it["message"]
                issues.append(issue_copy)
        except Exception as exc:
            it = issue(hdu_index, hdu_name, col_name, f"Error reading column: {str(exc)[:200]}", "INFO", release_label=release_label, category="error_reading_column")
            col_info["issues"] = [it]
            issues.append(it)

        columns.append(col_info)

    if nrows == 0 and not release_label:
        issues.append(issue(hdu_index, hdu_name, "", "Table has 0 rows", "CRITICAL", category="empty_table", pct=100.0))

    return {
        "index": hdu_index,
        "name": hdu_name,
        "type": type(hdu).__name__,
        "nrows": nrows,
        "ncols": len(hdu.columns) if hdu.columns is not None else None,
        "columns": columns,
        "issues": issues,
        "flag_statistics": flag_statistics,
        "release_label": release_label,
        "collapsed": nrows == 0 or release_label is not None,
    }


def analyze_fits_file(filepath, expected_max_mjd=None):
    # Walk the FITS file and build the report data structure.
    results = {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "filesize": os.path.getsize(filepath),
        "hdus": [],
        "all_issues": [],
        "error": None,
    }
    order = 0

    try:
        with fits.open(filepath, memmap=True) as hdul:
            results["n_hdus"] = len(hdul)
            total_hdus = len(hdul)

            for i, hdu in enumerate(hdul):
                hdu_info = {
                    "index": i,
                    "name": hdu.name,
                    "type": type(hdu).__name__,
                    "columns": [],
                    "issues": [],
                    "header_info": {},
                    "release_groups": [],
                }

                header = hdu.header
                for key in ["EXTNAME", "NAXIS1", "NAXIS2", "TFIELDS", "TELESCOP", "INSTRUME", "OBJECT"]:
                    if key in header:
                        hdu_info["header_info"][key] = str(header[key])

                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    # Table HDUs get the detailed quality checks and the flag summary.
                    try:
                        nrows = hdu.data.shape[0] if hdu.data is not None else 0
                    except Exception:
                        nrows = header.get("NAXIS2", 0)
                    hdu_info["nrows"] = int(nrows)
                    hdu_info["ncols"] = len(hdu.columns) if hdu.columns is not None else None
                    hdu_info["collapsed"] = (int(nrows) == 0)

                    base = analyze_hdu_table(hdu, i, total_hdus, expected_max_mjd=expected_max_mjd)
                    hdu_info["columns"] = base["columns"]
                    hdu_info["flag_statistics"] = base.get("flag_statistics", [])
                    hdu_info["issues"] = []
                    for it in base["issues"]:
                        it = dict(it)
                        it["order"] = order
                        order += 1
                        if it.get("release_label") is None:
                            results["all_issues"].append(it)
                            hdu_info["issues"].append(it)

                    # Release split only for HDU 2.
                    if i == 2 and hdu.data is not None and hdu.columns is not None and "release" in [c.name.lower() for c in hdu.columns]:
                        release_vals = np.asarray([normalize_value(v) for v in hdu.data["release"]], dtype=object)
                        seen = []
                        for rv in release_vals:
                            if rv and rv not in seen:
                                seen.append(rv)

                        for rv in seen:
                            mask = release_vals == rv
                            release_base = analyze_hdu_table(hdu, i, total_hdus, expected_max_mjd=expected_max_mjd, release_label=rv, release_mask=mask)
                            release_group = {
                                "release": rv,
                                "nrows": int(np.sum(mask)),
                                "columns": release_base["columns"],
                                "issues": [],
                                "collapsed": True,
                            }
                            for it in release_base["issues"]:
                                it = dict(it)
                                it["order"] = order
                                order += 1
                                release_group["issues"].append(it)
                                results["all_issues"].append(it)
                            hdu_info["release_groups"].append(release_group)

                else:
                    if hdu.data is not None and hasattr(hdu.data, "shape"):
                        shape = tuple(hdu.data.shape)
                        if len(shape) >= 1:
                            hdu_info["nrows"] = shape[0]
                        if len(shape) >= 2:
                            hdu_info["ncols"] = shape[1]

                results["hdus"].append(hdu_info)

    except Exception as exc:
        results["error"] = f"{exc}\n{traceback.format_exc()}"
        results["all_issues"].append(issue(-1, "", "", f"Failed to open file: {str(exc)}", "CRITICAL", category="file_open_error", pct=100.0))

    return results


def prune_release_specific_issues(results):
    # Remove release duplicates that already appear in the full file.
    # The full file is the baseline; release groups should only keep new problems.
    """Remove release issues that are already present in the full file."""
    for hdu in results.get("hdus", []):
        # Only HDU 2 has the release-level split.
        # The other HDUs keep their original full-table analysis only.
        if hdu.get("index") != 2:
            continue

        full_signatures = {it.get("signature") for it in hdu.get("issues", []) if it.get("signature") is not None}

        if not full_signatures:
            # Nothing to dedupe against; leave the release groups intact.
            break

        pruned = []
        for group in hdu.get("release_groups", []):
            # Keep only release-unique issues in these sections.
            kept_columns = []
            kept_issues = []
            for col in group.get("columns", []):
                col_issues = [it for it in col.get("issues", []) if it.get("signature") not in full_signatures]
                if not col_issues:
                    continue
                col2 = dict(col)
                col2["issues"] = col_issues
                kept_columns.append(col2)
                kept_issues.extend(col_issues)

            if kept_columns:
                group2 = dict(group)
                group2["columns"] = kept_columns
                group2["issues"] = kept_issues
                pruned.append(group2)

        hdu["release_groups"] = pruned

        # Also remove duplicate release issues from the top-level issue list.
        keep = []
        for it in results.get("all_issues", []):
            if it.get("release_label") is not None and it.get("signature") in full_signatures:
                continue
            keep.append(it)
        results["all_issues"] = keep
        break

    return results


def render_issue_html(issue):
    # Render the issue in the three-line layout.
    # The first line is the HDU/column key, the second line is the release, and the third is the message.
    parts = []
    prefix = f'HDU {issue["hdu_index"]} ({issue.get("hdu_name","")}), col "{issue.get("colname","")}":'
    parts.append(f'<div class="issue-entry">')
    parts.append(f'<div class="issue-prefix"><strong>{html.escape(prefix)}</strong></div>')
    if issue.get("release_label"):
        parts.append(f'<div class="issue-release">release {html.escape(str(issue["release_label"]))}</div>')
    body = html.escape(issue.get("message", ""))
    body = re.sub(r'(\d+(?:\.\d+)?)%', r'<strong>\1%</strong>', body)
    parts.append(f'<div class="issue-body">{body}</div>')
    parts.append('</div>')
    return "".join(parts)


def render_issue_block(issue, anchor_id=None):
    # Wrap a single issue with its severity badge.
    cls = issue.get("severity", "INFO")
    a = f' id="{anchor_id}"' if anchor_id else ""
    return f'<div{a} class="issue {cls}"><span class="badge {cls}">{cls}</span>{render_issue_html(issue)}</div>'


def issue_section_anchor(sev):
    # Map severity to the first anchor in that section.
    return {"CRITICAL": "critical-issues-start", "WARNING": "warning-issues-start", "INFO": "info-issues-start"}.get(sev, "issues-section")


def build_hdu_summary_bar(results):
    # Build the clickable HDU summary chips.
    chips = []
    for hdu in results.get("hdus", []):
        # Only HDU 2 has the release-level split.
        idx = hdu.get("index")
        name = hdu.get("name") or ""
        nrows = hdu.get("nrows")
        ncols = hdu.get("ncols")
        label = f"HDU {idx}: "
        if nrows is not None and ncols is not None:
            label += f"{nrows} rows, {ncols} cols"
        elif nrows is not None:
            label += f"{nrows} rows"
        elif ncols is not None:
            label += f"{ncols} cols"
        else:
            label += hdu.get("type", "")
        if name:
            label += f" ({name})"
        chips.append(f'<a class="file-chip" href="#hdu-table-controls">{html.escape(label)}</a>')
    return "".join(chips)


def render_filter_control():
    # Show the table filter above the detailed HDU tables.
    return """
  <div class="section filter-section" id="hdu-table-controls">
    <h2>Detailed HDU Tables</h2>
    <div class="filter-row">
      <label for="severity-filter"><strong>Show table rows with severity at least</strong></label>
      <select id="severity-filter" onchange="filterSeverity()">
        <option value="0">OK</option>
        <option value="1">Info</option>
        <option value="2">Warning</option>
        <option value="3">Critical</option>
      </select>
      <button type="button" class="reset-button" onclick="resetSeverityFilter()">Show all</button>
    </div>
  </div>
"""


def render_columns_table(columns, release_label=None):
    # Render one table of column stats and issues.
    # This is shared by the full HDU view and the release-specific view.
    if not columns:
        return '<p class="no-issues">No columns to display.</p>'

    out = ["""
    <table>
    <thead><tr>
      <th>#</th><th>Column</th><th>Format</th><th>Type</th><th>Unit</th>
      <th>Min</th><th>Max</th><th>Mean</th><th>Std</th>
      <th>NaN</th><th>Issues</th>
    </tr></thead>
    <tbody>
"""]
    for i, col in enumerate(columns):
        stats = col.get("stats", {})
        issues = col.get("issues", [])
        row_rank = max((SEVERITY_RANK.get(x.get("severity", "OK"), 0) for x in issues), default=0)
        row_cls = {0: "row-ok", 1: "row-info", 2: "row-warning", 3: "row-critical"}[row_rank]
        issue_html = "".join(render_issue_block(x) for x in sorted(issues, key=sort_key)) if issues else '<span class="no-issues">OK</span>'
        out.append(f"""      <tr class="{row_cls}" data-severity-rank="{row_rank}">
        <td>{i + 1}</td>
        <td class="mono">{html.escape(col.get('name', ''))}</td>
        <td class="mono">{html.escape(col.get('format', ''))}</td>
        <td class="mono">{html.escape(col.get('dtype', ''))}</td>
        <td>{html.escape(col.get('unit', ''))}</td>
        <td class="mono">{format_number(stats.get('min'))}</td>
        <td class="mono">{format_number(stats.get('max'))}</td>
        <td class="mono">{format_number(stats.get('mean'))}</td>
        <td class="mono">{format_number(stats.get('std'))}</td>
        <td class="mono">{format_number(stats.get('nan_count'))}</td>
        <td class="col-issues">{issue_html}</td>
      </tr>
""")
    out.append("    </tbody></table>\n")
    return "".join(out)



def render_flag_statistics(flag_statistics):
    # Render a nested dropdown with one table per flag column.
    if not flag_statistics:
        return ""

    parts = ["""
  <div class="section" style="margin-top:12px;">
    <details>
      <summary><strong>Flag Statistics</strong></summary>
"""]
    for flag_stat in flag_statistics:
        flag_name = html.escape(flag_stat["name"])
        total_rows = flag_stat["total_rows"]
        parts.append(f"""
      <details>
        <summary><strong>{flag_name}</strong> &mdash; {total_rows} rows</summary>
        <table>
          <thead>
            <tr>
              <th>Bit</th>
              <th>Rows with bit set</th>
              <th>Percent</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="mono" style="vertical-align: middle;">None (flag == 0)</td>
              <td>{flag_stat["zero_count"]}</td>
              <td>{flag_stat["zero_pct"]:.1f}%</td>
            </tr>
""")
        for bit_info in flag_stat["bits"]:
            parts.append(f"""
            <tr>
              <td>{bit_info["bit"]}</td>
              <td>{bit_info["count"]}</td>
              <td>{bit_info["pct"]:.1f}%</td>
            </tr>
""")
        parts.append("""
          </tbody>
        </table>
      </details>
""")
    parts.append("""
    </details>
  </div>
""")
    return "".join(parts)
def collapse_by_default(open_flag):
    # Keep the old open/closed decision in one place.
    return "" if open_flag else " open='false'"


def generate_html_report(results):
    # Assemble the final HTML report.
    # The page is built in sections so the anchors and collapse states stay predictable.
    filename = results["filename"]
    filesize_mb = results.get("filesize", 0) / (1024 * 1024)
    all_issues = sorted(results.get("all_issues", []), key=sort_key)
    summary_issues = sorted(top_level_issues(results), key=summary_sort_key)

    n_critical = sum(1 for x in summary_issues if x.get("severity") == "CRITICAL")
    n_warning = sum(1 for x in summary_issues if x.get("severity") == "WARNING")
    n_info = sum(1 for x in all_issues if x.get("severity") == "INFO")

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FITS QA Report: {html.escape(filename)}</title>
<style>
  html {{ scroll-behavior: smooth; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 1.5em; margin-bottom: 5px; }}
  h2 {{ font-size: 1.2em; margin: 20px 0 10px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
  h3 {{ font-size: 1.05em; margin: 15px 0 8px; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; }}
  .header .subtitle {{ color: #aaa; font-size: 0.9em; }}
  .file-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
  .file-chip {{ display: inline-block; text-decoration: none; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.12); color: white; border-radius: 999px; padding: 8px 12px; font-size: 0.9em; }}
  .file-chip:hover {{ background: rgba(255,255,255,0.2); }}
  .summary-cards {{ display: flex; gap: 15px; margin: 15px 0; flex-wrap: wrap; }}
  .card-link {{ text-decoration: none; color: inherit; flex: 1; min-width: 150px; display: block; }}
  .card {{ height: 100%; background: white; border-radius: 8px; padding: 15px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 150px; }}
  .card .number {{ font-size: 2em; font-weight: bold; }}
  .card .label {{ color: #666; font-size: 0.85em; text-transform: uppercase; }}
  .card.critical {{ border-left: 4px solid #dc3545; }}
  .card.critical .number {{ color: #dc3545; }}
  .card.warning {{ border-left: 4px solid #fd7e14; }}
  .card.warning .number {{ color: #fd7e14; }}
  .card.info {{ border-left: 4px solid #0d6efd; }}
  .card.info .number {{ color: #0d6efd; }}
  .card.neutral {{ border-left: 4px solid #6c757d; }}
  .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .filter-section {{ padding: 14px 20px; scroll-margin-top: 18px; }}
  .filter-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .filter-row select {{ padding: 6px 10px; border-radius: 6px; border: 1px solid #ccc; background: white; }}
  .reset-button {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #ccc; background: #f8f9fa; cursor: pointer; }}
  .reset-button:hover {{ background: #eef1f4; }}
  .issue {{ padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 0.9em; scroll-margin-top: 18px; }}
  .issue.CRITICAL {{ background: #f8d7da; border-left: 4px solid #dc3545; color: #842029; }}
  .issue.WARNING {{ background: #fff3cd; border-left: 4px solid #fd7e14; color: #664d03; }}
  .issue.INFO {{ background: #cff4fc; border-left: 4px solid #0d6efd; color: #055160; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.75em; font-weight: bold; color: white; margin-right: 8px; vertical-align: top; }}
  .badge.CRITICAL {{ background: #dc3545; }}
  .badge.WARNING {{ background: #fd7e14; }}
  .badge.INFO {{ background: #0d6efd; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; margin: 10px 0; }}
  th {{ background: #f8f9fa; text-align: left; padding: 8px; border-bottom: 2px solid #dee2e6; position: sticky; top: 0; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:hover {{ background: #f8f9fa; }}
  .mono {{ font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 0.9em; }}
  .col-issues {{ max-width: 520px; }}
  .no-issues {{ color: #198754; font-style: italic; }}
  .issue-entry {{ display: block; }}
  .issue-prefix {{ font-weight: 700; margin-bottom: 3px; }}
  .issue-release {{ font-weight: 400; margin-bottom: 3px; padding-left: 16px; }}
  .issue-body {{ padding-left: 16px; }}
  .row-ok {{}}
  .row-info {{ background: #f1fbff; }}
  .row-warning {{ background: #fff8e0; }}
  .row-critical {{ background: #fff0f0; }}
</style>
<script>
function filterSeverity() {{
  const select = document.getElementById('severity-filter');
  const selected = parseInt(select.value, 10);
  document.querySelectorAll('tr[data-severity-rank]').forEach((row) => {{
    const rowSeverity = parseInt(row.dataset.severityRank, 10);
    row.style.display = selected === 0 || rowSeverity >= selected ? '' : 'none';
  }});
}}
function resetSeverityFilter() {{
  const select = document.getElementById('severity-filter');
  if (select) {{
    select.value = '0';
  }}
  filterSeverity();
}}
</script>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{html.escape(filename)}</h1>
    <div class="subtitle">{html.escape(results['filepath'])}</div>
    <div class="subtitle">Size: {filesize_mb:.1f} MB | HDUs: {results.get('n_hdus', '?')}</div>
    <div class="file-bar">
      {build_hdu_summary_bar(results)}
    </div>
  </div>

  <div class="summary-cards">
    <a class="card-link" href="#critical-issues-start"><div class="card critical"><div class="number">{n_critical}</div><div class="label">Critical</div></div></a>
    <a class="card-link" href="#warning-issues-start"><div class="card warning"><div class="number">{n_warning}</div><div class="label">Warnings</div></div></a>
    <a class="card-link" href="#issues-section"><div class="card neutral"><div class="number">{n_critical + n_warning}</div><div class="label">Total Issues</div></div></a>
  </div>
"""]

    if results.get("error"):
        html_parts.append(f"""
  <div class="section">
    <h2>File Error</h2>
    <div class="issue CRITICAL"><pre>{html.escape(results['error'])}</pre></div>
  </div>
""")

    if summary_issues:
        html_parts.append('  <div class="section" id="issues-section">\n    <h2>All Issues (sorted by severity and impact)</h2>\n')
        seen = set()
        for it in summary_issues:
            sev = it.get("severity", "INFO")
            anchor = None
            if sev not in seen and sev in {"CRITICAL", "WARNING"}:
                anchor = issue_section_anchor(sev)
                seen.add(sev)
            html_parts.append(render_issue_block(it, anchor_id=anchor) + "\n")
        html_parts.append("  </div>\n")

    html_parts.append(render_filter_control())

    for hdu in results.get("hdus", []):
        # Only HDU 2 has the release-level split.
        idx = hdu["index"]
        hdu_label = f'HDU {idx}: {hdu.get("name","")} ({hdu.get("type","")})'
        nrows = hdu.get("nrows", 0)
        ncols = hdu.get("ncols")
        summary = []
        if nrows is not None:
            summary.append(f"{nrows} rows")
        if ncols is not None:
            summary.append(f"{ncols} columns")
        summary_text = ", ".join(summary)
        details_open = "" if hdu.get("collapsed") else " open"

        html_parts.append(f"""
  <div class="section" id="hdu-section-{idx}">
    <details{details_open}>
      <summary><h2 style="display:inline">{html.escape(hdu_label)}</h2>{(' &mdash; ' + html.escape(summary_text)) if summary_text else ''}</summary>
""")

        if hdu.get("header_info"):
            html_parts.append("    <p class=\"stats\">")
            for k, v in hdu["header_info"].items():
                html_parts.append(f'{html.escape(k)}={html.escape(str(v))} &nbsp; ')
            html_parts.append("</p>\n")

        if hdu.get("columns"):
            html_parts.append(render_columns_table(hdu["columns"]))
        else:
            html_parts.append('<p class="no-issues">No table columns.</p>')

        for group in hdu.get("release_groups", []):
            # Keep only release-unique issues in these sections.
            release = group.get("release", "")
            rows = group.get("nrows", 0)
            html_parts.append(f"""
      <details>
        <summary><strong>Release {html.escape(str(release))}</strong> &mdash; {rows} rows</summary>
""")
            html_parts.append(render_columns_table(group.get("columns", []), release_label=release))
            html_parts.append("      </details>\n")


        if hdu.get("flag_statistics") and nrows and int(nrows) > 0:
            html_parts.append(render_flag_statistics(hdu.get("flag_statistics", [])))
        html_parts.append("    </details>\n  </div>\n")

    html_parts.append("""
</div>
</body>
</html>
""")
    return "".join(html_parts)


def parse_args():
    # Parse the CLI arguments.
    parser = argparse.ArgumentParser(description="Analyze FITS files and generate HTML QA reports.")
    parser.add_argument("--summary-dir", default=SUMMARY_DIR, help="Directory containing FITS files to validate.")
    parser.add_argument("--save_dir", default=None, help="Directory to save the reports if different than the default validation directory ($MWM_ASTRA/{version}/validation).")
    parser.add_argument("--expected-max-mjd", type=float, default=None, help="Expected maximum MJD for columns ending in max_mjd.")
    return parser.parse_args()


def main():
    # Find files, analyze them, and write the reports.
    # Duplicate .fits / .fits.gz pairs are collapsed to one base filename.
    args = parse_args()
    summary_dir = args.summary_dir
    save_dir = args.save_dir if args.save_dir is not None else VALIDATION_DIR
    expected_max_mjd = args.expected_max_mjd

    os.makedirs(save_dir, exist_ok=True)

    fits_files = []
    seen_base = set()
    for f in os.listdir(summary_dir):
        if not (f.endswith(".fits") or f.endswith(".fits.gz")):
            continue
        base = f.replace(".fits.gz", "").replace(".fits", "")
        if base in seen_base:
            continue
        full_path = os.path.join(summary_dir, f)
        if os.path.isfile(full_path):
            fits_files.append(full_path)
            seen_base.add(base)

    fits_files.sort()
    print(f"Found {len(fits_files)} FITS files to analyze")
    if expected_max_mjd is not None:
        print(f"Using expected maximum MJD: {expected_max_mjd}")

    for filepath in fits_files:
        filename = os.path.basename(filepath)
        report_name = f"report_{filename.replace('.fits.gz', '.html').replace('.fits', '.html')}"
        report_path = os.path.join(save_dir, report_name)

        print("\n" + "=" * 60)
        print(f"Analyzing: {filename}")
        print(f"  Size: {os.path.getsize(filepath) / (1024 * 1024):.1f} MB")

        try:
            results = analyze_fits_file(filepath, expected_max_mjd=expected_max_mjd)
            prune_release_specific_issues(results)

            n_critical = sum(1 for x in results["all_issues"] if x.get("severity") == "CRITICAL")
            n_warning = sum(1 for x in results["all_issues"] if x.get("severity") == "WARNING")
            n_info = sum(1 for x in results["all_issues"] if x.get("severity") == "INFO")
            print(f"  Issues: {n_critical} critical, {n_warning} warnings, {n_info} info")

            html_report = generate_html_report(results)
            with open(report_path, "w") as f:
                f.write(html_report)
            print(f"  Report written: {report_path}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All reports generated.")


if __name__ == "__main__":
    main()
