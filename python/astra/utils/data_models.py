import os
import re


#https://github.com/sdss/datamodel_parser/blob/c19fb6a225946fb4189d3ff1d4e1a95b98f8b721/docs/flagship_datamodels.py
__dsi_path_descriptors = [
    # $APOGEE_REDUX/{APRED_VERS}/stars/{TELESCOPE}/{FIELD}/{PREFIX}Star-{APRED_VERS}-{OBJ}.fits
    ("apStar", r".+\/(?P<_APRED_VERS>[\w\d]+)\/stars\/(?P<TELESCOPE>\w{3}\d{1,2}m)\/(?P<FIELD>[\w\d\+\-]+)\/(?P<PREFIX>[\w\d]+)Star-(?P<APRED_VERS>[\w\d]+)-(?P<APOGEE_ID>[\w\d\+\-]+)\.fits?$"),

    # $APOGEE_REDUX/{APRED_VERS}/{TELESCOPE}/{PLATE}/{MJD}/apVisit-{PLATE}-{MJD}-{FIBER}.fits
    # https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/TELESCOPE/PLATE_ID/MJD5/apVisit.html
    ("apVisit", r".+\/(?P<_APRED_VERS>[\w\d]+)\/(?P<TELESCOPE>\w{3}\d{1,2}m)\/(?P<_PLATE>\d+)\/(?P<_MJD>\d+)\/apVisit-(?P<APRED_VERS>[\w\d]+)-(?P<PLATE>[0-9]{4})-(?P<MJD>[0-9]{5})-(?P<FIBER>[0-9]{3})\.fits?$"),

    # SDSSPath("sdss5").templates["apVisit"]
    # '$APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/apVisit-{apred}-{telescope}-{plate}-{mjd}-{fiber:0>3}.fits'    
    ("apVisit", r".+\/(?P<_APRED_VERS>.+)\/visit\/(?P<_TELESCOPE>\w{3}\d{1,2}m)\/(?P<FIELD>.+)\/(?P<_MJD>\d+)\/apVisit-(?P<APRED_VERS>.+)-(?P<TELESCOPE>\w{3}\d{1,2}m)-(?P<PLATE>[0-9]{4})-(?P<MJD>[0-9]{5})-(?P<FIBER>[0-9]{3})\.fits?$"),

    # SDSS-V
    #$APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits
    #$APOGEE_REDUX/daily/visit/apo25m/RM_XMM-LSS/15002/59146/apVisit-daily-apo25m-15002-59146-008.fits
    ("apVisit", r".+\/(?P<_APRED>[\w\d]+)\/visit\/(?P<_TELESCOPE>\w{3}\d{1,2}m)\/(?P<FIELD>.+)\/(?P<_PLATE>\d+)\/(?P<_MJD>\d+)\/(?P<PREFIX>a[s|p])Visit-(?P<APRED>[\w\d]+)-(?P<TELESCOPE>\w{3}\d{1,2}m)-(?P<PLATE>\d+)-(?P<MJD>\d+)-(?P<FIBER>[0-9]{3}).fits?$"),

    # $BOSS_SPECTRO_REDUX/{RUN2D}/spectra/{PLATE4}/spec-{PLATE}-{MJD}-{FIBER}.fits
    ("spec", r".+\/(?P<RUN2D>[\w\d_]+)\/spectra/(?P<PLATE4>\d+)\/spec-(?P<PLATE>\d+)-(?P<MJD>\d+)-(?P<FIBER>\d+)\.fits?$"),

    # https://data.sdss.org/datamodel/files/MANGA_SPECTRO_MASTAR/DRPVER/MPROCVER/mastar-goodspec-DRPVER-MPROCVER.html
    ("MaStar", r".+\/(?P<_DRPVER>[\w\d_]+)\/(?P<_MPROCVER>[\w\d_]+)/mastar-(?P<SUBSET>\w+)+-(?P<DRPVER>[\w\d_]+)-(?P<MPROCVER>[\w\d_]+).fits?$"),
]




def parse_descriptors(path, strict=True):
    r"""
    Parse the data model descriptors from the given path. For example, the path

    $APOGEE_REDUX/{apred}/stars/{telescope}/{field}/{prefix}Star-{apred}-{obj}.fits
    
    describes an apStar data model file of an observation of an object, taken with a particular
    telescope, towards some field, etc. This will parse a given path and return those data model 
    descriptors.

    :param path:
        A local path to a SDSS data product that has a data model registered with the SDSS Data
        Specification Index.

    :param strict: [optional]
        Require that the given path follows the full path description given by the data model,
        instead of just matching on the path basename (default: True).

    :returns:
        A two-length tuple containing the name of the matched data model, and a dictionary that
        contains the matched descriptors.

    :raises ValueError:
        If no data model could be found that describes the given path, or multiple data models were
        matched.
    """

    descriptors = lambda pattern, path: re.compile(pattern).search(path).groupdict()

    matches = dict()
    for name, strict_pattern in __dsi_path_descriptors:

        try:
            matches[name] = descriptors(strict_pattern, path)

        except AttributeError:
            if not strict:
                # Parse the strict pattern into something less strict.
                _, basename_pattern = strict_pattern.rsplit("\/", maxsplit=1)

                try:
                    matches[name] = descriptors(f"^{basename_pattern}", os.path.basename(path))

                except AttributeError:
                    pass

    if not len(matches):
        raise ValueError("no data model found that describe the given path")

    if len(matches) > 1:
        raise ValueError(f"multiple data model matches found: {matches}")

    return matches.popitem()


def parse_data_model(path, strict=True):
    r"""
    Return the SDSS data model that describes the given path.

    :param path:
        A local path to a SDSS data product that has a data model registered with the SDSS Data
        Specification Index.

    :param strict: [optional]
        Require that the given path follows the full path description given by the data model,
        instead of just matching on the path basename (default: True).

    :returns:
        The name of the matched data model.

    :raises ValueError:
        If no data model could be found that describes the given path, or multiple data models were
        matched.
    """

    return parse_descriptors(path, strict=strict)[0]


if __name__ == "__main__":

    paths = [
        "/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/r12/stars/apo25m/M67/apStar-r12-2M08485930+1117220.fits",
        "/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/r12/stars/apo25m/M67/apStar-r12-2M08485930+1117220.fits",
        "/APOGEE_REDUX/r8/apo25m/4912/55726/apVisit-r8-4912-55726-269.fits",
        "/Users/arc/Downloads/spec-6171-56311-0312.fits",
    ]

    for path in paths:
        try:
            print(parse_descriptors(path, strict=False))

        except:
            raise

    print("STRICT")
    for path in paths:
        try:
            print(parse_descriptors(path, strict=True))

        except:
            print(f"Couldn't match path {path} on strict mode")



