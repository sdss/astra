from astropy.io import fits
import numpy as np

import psutil

    
def parse_boss_visit_meta(hdu, source_meta):

    if hdu.header["DATASUM"] == "0":
        return []
    
    observatory = hdu.header["OBSRVTRY"]
    telescope = observatory.lower() + "25m"
    common = source_meta.copy()
    common.update(
        observatory=observatory,
        instrument=hdu.header["INSTRMNT"],
        telescope=telescope,
        v_boss=hdu.header["V_BOSS"],
        nres=hdu.header["NRES"],
        filtsize=hdu.header["FILTSIZE"],
        normsize=hdu.header["NORMSIZE"],
        conscale=hdu.header["CONSCALE"],        
    )    

    keys = (
        "VJAEGER",
        "VKAIJU",
        "VCOORDIO",
        "VCALIBS",
        "VERSIDL",
        "VERSUTIL",
        "VERSREAD",
        "VERS2D",
        "VERSCOMB",
        "VERSLOG",
        "VERSFLAT",
        "DIDFLUSH",
        "CARTID",
        "PSFSKY",
        "PREJECT",
        "LOWREJ",
        "HIGHREJ",
        "SCATPOLY",
        "PROFTYPE",
        "NFITPOLY",
        "SKYCHI2",
        "SCHI2MIN",
        "SCHI2MAX",
        "V_HELIO",
        "RDNOISE0"
    )
    common.update({key.lower(): hdu.header[key] for key in keys})

    ignore = (
        "flux",
        "e_flux",
        "bitmask",
        "continuum",
        "wresl",
        "data_product_id"
    )
    N = len(hdu.data)
    visits = []
    for i in range(N):
        visit_meta = dict(**common, hdu_data_index=i)
        for name in hdu.data.dtype.names:
            if name.lower() in ignore: continue
            visit_meta[name.lower()] = np.copy(hdu.data[name][i])
        visits.append(visit_meta)
    return visits



def parse_apogee_visit_meta(hdu, source_meta):

    if hdu.header["DATASUM"] == "0":
        return []

    observatory = hdu.header["OBSRVTRY"]
    telescope = observatory.lower() + "25m"
    common = source_meta.copy()
    common.update(
        observatory=observatory,
        instrument=hdu.header["INSTRMNT"],
        telescope=telescope,
        v_apred=hdu.header["V_APRED"],
        nres=hdu.header["NRES"],
        filtsize=hdu.header["FILTSIZE"],
        normsize=hdu.header["NORMSIZE"],
        conscale=hdu.header["CONSCALE"],        
    )

    common.update(
        teff_doppler=hdu.header["TEFF_D"],
        e_teff_doppler=hdu.header["E_TEFF_D"],
        logg_doppler=hdu.header["LOGG_D"],
        e_logg_doppler=hdu.header["E_LOGG_D"],
        fe_h_doppler=hdu.header["FEH_D"],
        e_fe_h_doppler=hdu.header["E_FEH_D"],
    )
    for key, value, comment in hdu.header.cards:
        if f"{value}".upper().startswith("DOPPLER STELLAR PARAMETERS"):
            v_doppler = value.split("(")[1].rstrip(")")
            break
    else:
        v_doppler = "???????"
    common.update(v_doppler=v_doppler)

    N = len(hdu.data)
    ignore = (
        "flux",
        "e_flux",
        "bitmask",
        "continuum",
        "data_product_id"
    )
    visits = []
    for i in range(N):
        visit_meta = dict(**common, hdu_data_index=i)
        for name in hdu.data.dtype.names:
            if name.lower() in ignore: continue
            visit_meta[name.lower()] = np.copy(hdu.data[name][i])
        visits.append(visit_meta)
    return visits


def calculate_star_rv(visits):

    # Calculate star-level RV from visits in the same way that
    # the APOGEE-DRP does, even if I think that way is wrong.
    v_rads = np.array([visit["v_rad"] for visit in visits])
    in_stack = np.array([visit["in_stack"] for visit in visits])
    snr = np.array([visit["snr"] for visit in visits])

    if any(in_stack):
        v_rad = np.sum(v_rads[in_stack] * snr[in_stack])/np.sum(snr[in_stack])
        v_scatter = np.std(v_rads[in_stack])
        v_err = v_scatter/np.sqrt(np.sum(in_stack))

        return (v_rad, v_scatter, v_err)
    else:
        return (np.nan, np.nan, np.nan)


def parse_meta(data_product):

    common_meta = {}

    with fits.open(data_product.path) as image:
        primary, *extensions = image

        common_meta.update(
            v_astra=primary.header["V_ASTRA"],
            created=primary.header["CREATED"],
            healpix=primary.header["HEALPIX"],
            gaia_source_id=primary.header["GAIA_ID"],
            # get gaia DR version
            gaia_data_release = primary.header.comments["GAIA_ID"].split(" ")[1],
            tic_id=primary.header["TIC_ID"],
            cat_id=primary.header["CAT_ID"],
            cat_id05=primary.header["CAT_ID05"],
            cat_id10=primary.header["CAT_ID10"],

            # astrometry
            ra=primary.header["RA"],
            dec=primary.header["DEC"],
            gaia_ra=primary.header["GAIA_RA"],
            gaia_dec=primary.header["GAIA_DEC"],
            plx=primary.header["PLX"],
            pmra=primary.header["PMRA"],
            pmde=primary.header["PMDE"],
            e_pmra=primary.header["E_PMRA"],
            e_pmde=primary.header["E_PMDE"],
            gaia_v_rad=primary.header["V_RAD"],
            gaia_e_v_rad=primary.header["E_V_RAD"],
            g_mag=primary.header["G_MAG"],
            bp_mag=primary.header["BP_MAG"],
            rp_mag=primary.header["RP_MAG"],
            j_mag=primary.header["J_MAG"],
            h_mag=primary.header["H_MAG"],
            k_mag=primary.header["K_MAG"],
            e_j_mag=primary.header["E_J_MAG"],
            e_h_mag=primary.header["E_H_MAG"],
            e_k_mag=primary.header["E_K_MAG"],

            # carton information?
            carton_0=primary.header["CARTON_0"],
            cartons=primary.header["CARTONS"],
            programs=primary.header["PROGRAMS"],
            mappers=primary.header["MAPPERS"],
            v_xmatch=primary.header["V_XMATCH"],
            
            mwmvisit_data_product_id=data_product.id,
            
        )

        boss_apo, boss_lco, apogee_apo, apogee_lco = extensions
        
        boss_visits = []
        for hdu in (boss_apo, boss_lco):
            boss_visits.extend(parse_boss_visit_meta(hdu, common_meta))

        apogee_visits = []
        for hdu in (apogee_apo, apogee_lco):
            apogee_visits.extend(parse_apogee_visit_meta(hdu, common_meta))

        # Prefer APOGEE RVs if we have them.    
        v_rad, v_scatter, v_err = calculate_star_rv(apogee_visits)
        if not np.isfinite(v_rad):
            v_rad, v_scatter, v_err = calculate_star_rv(boss_visits)

        visits = []
        visits.extend(boss_visits)
        visits.extend(apogee_visits)

        star = common_meta.copy()
        star.update(
            v_rad=v_rad,
            v_err=v_err,
            v_scatter=v_scatter,
        )

    del image

    return (visits, star)


if __name__ == "__main__":
    import os

    output_file = "mwm_visit_meta.pickle"

    if not os.path.exists(output_file):
            
        from astra.database.astradb import DataProduct
        from tqdm import tqdm

        q = (
            DataProduct
            .select()
            .where(DataProduct.filetype == "mwmVisit")
        ) 
        content = []
        for data_product in tqdm(q):
            visits, star = parse_meta(data_product)
            content.append((visits, star))
            #proc = psutil.Process()
            #print(len(proc.open_files())) 

        import pickle
        with open(output_file, "wb") as f:
            pickle.dump(content, f)

    else:
        import pickle
        with open(output_file, "rb") as fp:
            content = pickle.load(fp)

        # Let's create the schema.
        # TODO: this is so hacky, we should define it earlier.

        from astra.database.astradb import BaseModel
        from peewee import AutoField, BigIntegerField, IntegerField, TextField, FloatField, ForeignKeyField, BooleanField, DateTimeField
        from playhouse.postgres_ext import ArrayField

        dtypes = {}
        for visits, star in tqdm(content):
            for visit in visits:
                for key, value in visit.items():
                    dtype = type(value)
                    dtypes.setdefault(key, {})
                    dtypes[key].setdefault(dtype, 0)
                    dtypes[key][dtype] += 1
            
            for key, value in star.items():
                dtype = type(value)
                dtypes.setdefault(key, {})
                dtypes[key].setdefault(dtype, 0)
                dtypes[key][dtype] += 1
        
        prescribed_dtypes = {
            "telescope": "str"
        }
        most_common_dtype = {}
        for key, key_dtypes in dtypes.items():
            if key in prescribed_dtypes:
                continue
            possible_dtypes = set(key_dtypes.keys()).difference({type(None)})
            sorted_dtypes = sorted(key_dtypes.items(), key=lambda x: x[1], reverse=True)
            for dtype, count in sorted_dtypes:
                if dtype == type(None):
                    continue
                most_common_dtype[key] = dtype
                print(key, dtype)
                break

        
        class StarMeta(BaseModel):
            pk = AutoField()
            
            astra_version_major = IntegerField()
            astra_version_minor = IntegerField()
            astra_version_patch = IntegerField()

            created = DateTimeField()

            healpix = IntegerField()
            gaia_source_id = BigIntegerField(null=True)
            gaia_data_release = TextField(null=True)

            cat_id = BigIntegerField()
            tic_id = BigIntegerField(null=True)
            cat_id05 = BigIntegerField(null=True)
            cat_id10 = BigIntegerField(null=True)

            ra = FloatField()
            dec = FloatField()
            gaia_ra = FloatField(null=True)
            gaia_dec = FloatField(null=True)
            plx = FloatField(null=True)
            pmra = FloatField(null=True)
            pmde = FloatField(null=True)
            e_pmra = FloatField(null=True)
            e_pmde = FloatField(null=True)
            gaia_v_rad = FloatField(null=True)
            gaia_e_v_rad = FloatField(null=True)
            g_mag = FloatField(null=True)
            bp_mag = FloatField(null=True)
            rp_mag = FloatField(null=True)
            j_mag = FloatField(null=True)
            h_mag = FloatField(null=True)
            k_mag = FloatField(null=True)
            e_j_mag = FloatField(null=True)
            e_h_mag = FloatField(null=True)
            e_k_mag = FloatField(null=True)
            
            carton_0 = TextField(null=True)
            v_xmatch = TextField()

            # Doppler results.
            doppler_teff = FloatField(null=True)
            doppler_e_teff = FloatField(null=True)
            doppler_logg = FloatField(null=True)
            doppler_e_logg = FloatField(null=True)
            doppler_fe_h = FloatField(null=True)
            doppler_e_fe_h = FloatField(null=True)
            doppler_starflag = IntegerField(null=True)
            doppler_version = TextField(null=True)
            doppler_v_rad = FloatField(null=True)
            
            # The RXC SAO results are done per visit, not per star.
            # For convenience we include them here, but we will take
            # The result with the highest S/N.
            
            xcsao_teff = FloatField(null=True)
            xcsao_e_teff = FloatField(null=True)
            xcsao_logg = FloatField(null=True)
            xcsao_e_logg = FloatField(null=True)
            # TODO: Naming of this in files is feh
            xcsao_fe_h = FloatField(null=True)
            xcsao_e_fe_h = FloatField(null=True)            
            xcsao_rxc = FloatField(null=True)
            xcsao_v_rad = FloatField(null=True)
            xcsao_e_v_rad = FloatField(null=True)


        
        from astra.database.astradb import database
        from datetime import datetime

        database.create_tables([StarMeta])

        fields = [
            StarMeta.astra_version_major,
            StarMeta.astra_version_minor,
            StarMeta.astra_version_patch,
            StarMeta.created,
            StarMeta.healpix,
            StarMeta.gaia_source_id,
            StarMeta.gaia_data_release,
            StarMeta.cat_id,
            StarMeta.tic_id,
            StarMeta.cat_id05,
            StarMeta.cat_id10,
            StarMeta.ra,
            StarMeta.dec,
            StarMeta.gaia_ra,
            StarMeta.gaia_dec,
            StarMeta.plx,
            StarMeta.pmra,
            StarMeta.pmde,
            StarMeta.e_pmra,
            StarMeta.e_pmde,
            StarMeta.gaia_v_rad,
            StarMeta.gaia_e_v_rad,
            StarMeta.g_mag,
            StarMeta.bp_mag,
            StarMeta.rp_mag,
            StarMeta.j_mag,
            StarMeta.h_mag,
            StarMeta.k_mag,
            StarMeta.e_j_mag,
            StarMeta.e_h_mag,
            StarMeta.e_k_mag,
            
            StarMeta.carton_0,
            StarMeta.v_xmatch,

            StarMeta.doppler_teff,
            StarMeta.doppler_e_teff,
            StarMeta.doppler_logg,
            StarMeta.doppler_e_logg,
            StarMeta.doppler_fe_h,
            StarMeta.doppler_e_fe_h,
            StarMeta.doppler_starflag,
            StarMeta.doppler_version,
            StarMeta.doppler_v_rad,
            
            StarMeta.xcsao_rxc,
            StarMeta.xcsao_teff,
            StarMeta.xcsao_e_teff,
            StarMeta.xcsao_logg,
            StarMeta.xcsao_e_logg,
            StarMeta.xcsao_fe_h,
            StarMeta.xcsao_e_fe_h,
            StarMeta.xcsao_v_rad,
            StarMeta.xcsao_e_v_rad,
                        
        ]

        data = []
        datetime_fmt = "%y-%m-%d %H:%M:%S"
        for visits, star in tqdm(content):
            
            astra_version_major, astra_version_minor, astra_version_patch = tuple(map(int, star["v_astra"].split(".")))
            special = dict(
                astra_version_major=astra_version_major,
                astra_version_minor=astra_version_minor,
                astra_version_patch=astra_version_patch,
                created=datetime.strptime(star["created"], datetime_fmt),
                e_v_rad=star["v_err"]
            )
            # Get doppler teff from any of the visits.
            for visit in visits:                
                if "teff_doppler" in visit and visit["teff_doppler"]:
                    special.update(
                        doppler_teff=visit["teff_doppler"],
                        doppler_e_teff=visit["e_teff_doppler"],
                        doppler_logg=visit["logg_doppler"],
                        doppler_e_logg=visit["e_logg_doppler"],
                        doppler_fe_h=visit["fe_h_doppler"],
                        doppler_e_fe_h=visit["e_fe_h_doppler"],
                        doppler_version=visit["v_doppler"],
                        doppler_starflag=visit["starflag"],
                        doppler_v_rad=visit["v_rad"],
                    )
                    break
            
            # Get XCSAO from the highest S/N visit.
            highest_snr = None
            for visit in visits:
                if "teff_xcsao" in visit:
                    snr = np.atleast_2d(visit["snr"])[0]
                    if highest_snr is None or snr > highest_snr:
                        special.update(
                            xcsao_teff=visit["teff_xcsao"],
                            xcsao_e_teff=visit["e_teff_xcsao"],
                            xcsao_logg=visit["logg_xcsao"],
                            xcsao_e_logg=visit["e_logg_xcsao"],
                            xcsao_fe_h=visit["feh_xcsao"], # NOTE the missing underscore
                            xcsao_e_fe_h=visit["e_feh_xcsao"], # NOTE the missing underscore
                            xcsao_rxc=visit["rxc_xcsao"],
                            xcsao_v_rad=visit["v_rad"],
                            xcsao_e_v_rad=visit["e_v_rad"]
                        )
                        highest_snr = snr
            
            row = []
            for field in fields:
                if field.name in special:
                    row.append(special[field.name])
                else:
                    row.append(star.get(field.name, None))

            data.append(row)


        from peewee import chunked

        with database.atomic():
            for batch in tqdm(chunked(data[10000:], 10000)):
                StarMeta.insert_many(batch, fields=fields).execute()


        class VisitMeta(BaseModel):
            pk = AutoField()
            
            astra_version_major = IntegerField()
            astra_version_minor = IntegerField()
            astra_version_patch = IntegerField()

            created = DateTimeField()

            healpix = IntegerField()
            gaia_source_id = BigIntegerField(null=True)
            gaia_data_release = TextField(null=True)

            cat_id = BigIntegerField()
            tic_id = BigIntegerField(null=True)
            cat_id05 = BigIntegerField(null=True)
            cat_id10 = BigIntegerField(null=True)

            ra = FloatField()
            dec = FloatField()
            gaia_ra = FloatField(null=True)
            gaia_dec = FloatField(null=True)
            plx = FloatField(null=True)
            pmra = FloatField(null=True)
            pmde = FloatField(null=True)
            e_pmra = FloatField(null=True)
            e_pmde = FloatField(null=True)
            gaia_v_rad = FloatField(null=True)
            gaia_e_v_rad = FloatField(null=True)
            g_mag = FloatField(null=True)
            bp_mag = FloatField(null=True)
            rp_mag = FloatField(null=True)
            j_mag = FloatField(null=True)
            h_mag = FloatField(null=True)
            k_mag = FloatField(null=True)
            e_j_mag = FloatField(null=True)
            e_h_mag = FloatField(null=True)
            e_k_mag = FloatField(null=True)
            
            carton_0 = TextField(null=True)
            v_xmatch = TextField()


            # File type stuff
            release = TextField()
            filetype = TextField()
            plate = IntegerField(null=True)
            fiber = IntegerField(null=True)
            field = TextField(null=True)
            apred = TextField(null=True)
            prefix = TextField(null=True)
            mjd = IntegerField(null=True)            

            run2d = TextField(null=True)
            fieldid = TextField(null=True)
            isplate = TextField(null=True)
            catalogid = BigIntegerField(null=True)

            # Common stuff.
            observatory = TextField()
            instrument = TextField()
            hdu_data_index = IntegerField()
            snr = FloatField()
            fps = FloatField()
            in_stack = BooleanField()
            v_shift = FloatField(null=True)

            continuum_theta = ArrayField(FloatField)

            # APOGEE-level stuff.
            v_apred = TextField(null=True)
            nres = ArrayField(FloatField)
            filtsize = IntegerField()
            normsize = IntegerField()
            conscale = BooleanField()

            # Doppler results.
            doppler_teff = FloatField(null=True)
            doppler_e_teff = FloatField(null=True)
            doppler_logg = FloatField(null=True)
            doppler_e_logg = FloatField(null=True)
            doppler_fe_h = FloatField(null=True)
            doppler_e_fe_h = FloatField(null=True)
            doppler_starflag = IntegerField(null=True)
            doppler_version = TextField(null=True)         
         
            date_obs = DateTimeField(null=True)
            exptime = FloatField(null=True)
            fluxflam = FloatField(null=True)
            npairs = IntegerField(null=True)
            dithered = FloatField(null=True)

            jd = FloatField(null=True)
            v_rad = FloatField(null=True)
            e_v_rad = FloatField(null=True)
            v_rel = FloatField(null=True)
            v_bc = FloatField(null=True)
            rchisq = FloatField(null=True)
            n_rv_components = IntegerField(null=True)

            visit_pk = BigIntegerField(null=True)
            rv_visit_pk = BigIntegerField(null=True)

            v_boss = TextField(null=True)
            vjaeger = TextField(null=True)
            vkaiju = TextField(null=True)
            vcoordio = TextField(null=True)
            vcalibs = TextField(null=True)
            versidl = TextField(null=True)
            versutil = TextField(null=True)
            versread = TextField(null=True)
            vers2d = TextField(null=True)
            verscomb = TextField(null=True)
            verslog = TextField(null=True)
            versflat = TextField(null=True)
            didflush = BooleanField(null=True)
            cartid = TextField(null=True)
            psfsky = IntegerField(null=True)
            preject = FloatField(null=True)
            lowrej = IntegerField(null=True)
            highrej = IntegerField(null=True)
            scatpoly = IntegerField(null=True)
            proftype = IntegerField(null=True)
            nfitpoly = IntegerField(null=True)
            skychi2 = FloatField(null=True)
            schi2min = FloatField(null=True)
            schi2max = FloatField(null=True)
            rdnoise0 = FloatField(null=True)

            alt = FloatField(null=True)
            az = FloatField(null=True)
            seeing = FloatField(null=True)
            airmass = FloatField(null=True)
            airtemp = FloatField(null=True)
            dewpoint = FloatField(null=True)
            humidity = FloatField(null=True)
            pressure = FloatField(null=True)
            gustd = FloatField(null=True)
            gusts = FloatField(null=True)
            windd = FloatField(null=True)
            winds = FloatField(null=True)
            moon_dist_mean = FloatField(null=True)
            moon_phase_mean = FloatField(null=True)
            nexp = IntegerField(null=True)
            nguide = IntegerField(null=True)
            tai_beg = DateTimeField(null=True)
            tai_end = DateTimeField(null=True)
            fiber_offset = BooleanField(null=True)
            delta_ra = FloatField(null=True)
            delta_dec = FloatField(null=True)
            zwarning = IntegerField(null=True)

            xcsao_teff = FloatField(null=True)
            xcsao_e_teff = FloatField(null=True)
            xcsao_logg = FloatField(null=True)
            xcsao_e_logg = FloatField(null=True)
            # TODO: Naming of this in files is feh
            xcsao_fe_h = FloatField(null=True)
            xcsao_e_fe_h = FloatField(null=True)            
            xcsao_rxc = FloatField(null=True)

            #v_bc? = FloatField(null=True) # from v_helio?
    
        fields = list(VisitMeta._meta.fields.values())[1:]


        data = []
        datetime_fmt = "%y-%m-%d %H:%M:%S"
        item = lambda _: np.atleast_1d(_)[0]

        for visits, star in tqdm(content):
            for visit in visits:
                    
                astra_version_major, astra_version_minor, astra_version_patch = tuple(map(int, visit["v_astra"].split(".")))
                nguide = item(visit.get("nguide", None))
                if nguide is not None and not np.isfinite(nguide):
                    nguide = None
                special = dict(
                    astra_version_major=astra_version_major,
                    astra_version_minor=astra_version_minor,
                    astra_version_patch=astra_version_patch,
                    created=datetime.strptime(visit["created"], datetime_fmt),
                    nres=list(map(float, visit["nres"].split(" "))),
                    nguide=nguide
                )

                if "teff_doppler" in visit and visit["teff_doppler"]:
                    special.update(
                        doppler_teff=item(visit["teff_doppler"]),
                        doppler_e_teff=item(visit["e_teff_doppler"]),
                        doppler_logg=item(visit["logg_doppler"]),
                        doppler_e_logg=item(visit["e_logg_doppler"]),
                        doppler_fe_h=item(visit["fe_h_doppler"]),
                        doppler_e_fe_h=item(visit["e_fe_h_doppler"]),
                        doppler_version=item(visit["v_doppler"]),
                        doppler_starflag=item(visit["starflag"]),
                    )
            
                if "teff_xcsao" in visit:
                    special.update(
                        xcsao_teff=item(visit["teff_xcsao"]),
                        xcsao_e_teff=item(visit["e_teff_xcsao"]),
                        xcsao_logg=item(visit["logg_xcsao"]),
                        xcsao_e_logg=item(visit["e_logg_xcsao"]),
                        xcsao_fe_h=item(visit["feh_xcsao"]), # NOTE the missing underscor)e
                        xcsao_e_fe_h=item(visit["e_feh_xcsao"]), # NOTE the missing underscor)e
                        xcsao_rxc=item(visit["rxc_xcsao"]),
                    )
                
                row = []
                for field in fields:
                    if field.name in special:
                        value = special[field.name]
                    else:
                        value = visit.get(field.name, None)

                    if isinstance(value, (np.ndarray, )):
                        if field.name in ("continuum_theta", "nres"):
                            value = list(map(float, value))
                        else:
                            value = np.atleast_1d(value)[0]
                    
                    #if isinstance(value, (np.int, )):
                    #    value = int(value)
                    #el
                    #if isinstance(value, (np.float, np.float32, np.float64)):
                    #    value = float(value)
                    
                    '''
                    if isinstance(value, np.str_):
                        value = str(value)
                    elif isinstance(value, (np.bool_, )):
                        value = bool(value)
                    '''
                    
                    row.append(value)                 
                data.append(row)


        from peewee import chunked

        with database.atomic():
            for batch in tqdm(chunked(data, 10000)):
                VisitMeta.insert_many(batch, fields=fields).execute()
    
        raise a