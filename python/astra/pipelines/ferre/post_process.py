

def _post_process_ferre(pwd, model, data_product=None, ferre_timeout=None):
    
    stdout_path, stderr_path = _stdout_stderr_paths(pwd)

    with open(stdout_path, "r") as fp:
        stdout = fp.read()
    with open(stderr_path, "r") as fp:
        stderr = fp.read()

    n_done, n_error, control_kwds, meta = utils.parse_ferre_output(pwd, stdout, stderr)

    log.info(f"Found {n_done} completed successfully and {n_error} errors")
    
    # Parse outputs.
    path = os.path.join(pwd, control_kwds["PFILE"])
    input_names = np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str))

    # FFILE and ERFILE are inputs, so they will always be the right shape.
    try:
        path = os.path.join(pwd, control_kwds["FFILE"])
        flux = np.atleast_2d(np.loadtxt(path))
    except:
        log.exception(f"Failed to load input flux from {path}")
    try:
        path = os.path.join(pwd, control_kwds["ERFILE"])
        flux_sigma = np.atleast_2d(np.loadtxt(path))
    except:
        log.exception(f"Failed to load flux sigma from {path}")
        raise

    # Now parse the outputs from the FERRE run.
    path = os.path.join(pwd, control_kwds["OPFILE"])
    try:
        output_names, output_params, output_param_errs, meta = utils.read_output_parameter_file(
            path,
            n_dimensions=control_kwds["NDIM"],
            full_covariance=control_kwds["COVPRINT"],
        )
    except:
        log.exception(f"Exception when parsing FERRE output parameter file {path}")
        raise

    if len(input_names) > len(output_names):
        log.warning(f"Number of input parameters does not match output parameters ({len(input_names)} > {len(output_names)}). FERRE may have failed. We will pick up the pieces..")

    # Which entries are missing?
    missing_names = list(set(input_names).difference(output_names))
    missing_indices = [np.where(input_names == mn)[0][0] for mn in missing_names]
    for i in np.argsort(missing_indices):
        missing_name, missing_index = (missing_names[i], missing_indices[i])
        log.warning(f"Missing parameters for spectrum named {missing_name} (index {missing_index}; row {missing_index+1})")        

    # We will fill the missing parameters with nans, and missing fluxes with nans too
    N, P = flux.shape
    D = int(control_kwds["NDIM"]) 
    params = np.nan * np.ones((N, D), dtype=float)
    param_errs = np.nan * np.ones((N, D), dtype=float)
    log_chisq_fit = np.nan * np.ones(N)
    log_snr_sq = np.nan * np.ones(N)
    frac_phot_data_points = np.nan * np.ones(N)

    indices = []
    for i, name in enumerate(output_names):
        index, = np.where(input_names == name)
        assert len(index) == 1, f"Name {name} (index {i}) appears more than once in the input parameter file!"
        indices.append(index[0])
    indices = np.array(indices)

    params[indices] = output_params
    param_errs[indices] = output_param_errs
    log_chisq_fit[indices] = meta["log_chisq_fit"]
    log_snr_sq[indices] = meta["log_snr_sq"]
    frac_phot_data_points[indices] = meta["frac_phot_data_points"]


    offile_path = os.path.join(pwd, control_kwds["OFFILE"])
    _model_flux = np.atleast_2d(np.loadtxt(offile_path, usecols=1 + np.arange(P)))
    _model_flux_names = np.atleast_1d(np.loadtxt(offile_path, usecols=(0, ), dtype=str))
    model_indices = []
    for i, name in enumerate(_model_flux_names):
        index, = np.where(input_names == name)
        model_indices.append(index[0])
    model_indices = np.array(model_indices)

    model_flux = np.nan * np.ones((N, P), dtype=float)
    model_flux[model_indices] = _model_flux

    if "SFFILE" in control_kwds:
        try:
            sffile_path = os.path.join(pwd, control_kwds["SFFILE"])
            _normalized_flux = np.atleast_2d(np.loadtxt(sffile_path, usecols=1 + np.arange(P)))
            _normalized_flux_names = np.atleast_1d(np.loadtxt(sffile_path, usecols=(0, ), dtype=str))
        except:
            log.exception(f"Failed to load normalized observed flux from {sffile_path}")
            raise
        else:
            # Order the normalized flux to be the same as the inputs
            normalized_flux_indices = []
            for i, name in enumerate(_normalized_flux_names):
                index, = np.where(input_names == name)
                normalized_flux_indices.append(index[0])
            normalized_flux_indices = np.array(normalized_flux_indices)

            normalized_flux = np.nan * np.ones((N, P), dtype=float)
            normalized_flux[normalized_flux_indices] = _normalized_flux

            continuum = flux / normalized_flux
    else:
        continuum = np.ones_like(flux)
        normalized_flux = flux

    has_complete_results = (
        np.any(np.isfinite(params), axis=1)
    *   np.any(np.isfinite(model_flux), axis=1)
    )

    fill_value = -999
    # If we only have some things (eg params but no model flux) we should make it all nan,
    # ebcause we dont want to rely on this downstream
    params[~has_complete_results] = fill_value
    model_flux[~has_complete_results] = fill_value
    normalized_flux[~has_complete_results] = fill_value
    continuum[~has_complete_results] = fill_value
    
    header_path = control_kwds["SYNTHFILE(1)"]
    headers, *segment_headers = utils.read_ferre_headers(
        utils.expand_path(header_path)
    )
    parameter_names = utils.sanitise(headers["LABEL"])

    # Flag things.
    param_bitmask = bitmask.ParamBitMask()
    param_bitmask_flags = np.zeros(params.shape, dtype=np.int64)

    bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
    bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
    param_bitmask_flags[
        (params < bad_lower) | (params > bad_upper)
    ] |= param_bitmask.get_value("GRIDEDGE_BAD")

    warn_lower = headers["LLIMITS"] + headers["STEPS"]
    warn_upper = headers["ULIMITS"] - headers["STEPS"]
    param_bitmask_flags[
        (params < warn_lower) | (params > warn_upper)
    ] |= param_bitmask.get_value("GRIDEDGE_WARN")
    param_bitmask_flags[
        (params == fill_value) | (param_errs < -0.01) | ~np.isfinite(params)
    ] |= param_bitmask.get_value("FERRE_FAIL")
    
    # Check for any erroneous outputs
    if np.any(param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")):
        v = param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")
        idx = np.where(
            np.any(
                param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL"), axis=1
            )
        )
        log.warning(f"FERRE returned all erroneous values for an entry: {idx} {v}")

    # We need to yield N times, where N is len(data_products) == len(hdu), etc.
    # So we need to link things by (i, k) where i is the index of the data product
    # and k is the index of the segment.
    si = 0
    ferre_n_obj = params.shape[0]
    
    try:
        timings = utils.get_processing_times(stdout)
        ferre_times_elapsed = timings["time_per_spectrum"][indices]
        ferre_time_load = timings["time_load"]
    except:
        ferre_time_load = None
        ferre_times_elapsed = cycle([None])

    # Get data products if necessary.
    log.info(f"Collecting data products")
    if data_product is None:
        data_product_ids = list(set(map(int, (ea.split("_")[2] for ea in input_names))))
        dps = (
            DataProduct
            .select()
            .where(DataProduct.id << data_product_ids)
        )
        data_product_dict = {}
        for data_product in dps:
            data_product_dict[data_product.id] = data_product
    else:
        data_product_dict = {}
        for data_product in data_product:
            data_product_dict[data_product.id] = data_product

    # Load spectra.
    log.info(f"Loading spectra")
    spectra = {}
    for input_name in input_names:
        z, k, data_product_id, hdu = map(int, input_name.split("_"))

        key = (data_product_id, hdu)
        if key not in spectra: 
            spectra[key] = SpectrumList.read(data_product_dict[data_product_id].path, hdu=hdu)
    
    log.info("Yield results")

    # outputs must be per data_product_id, hdu
    # TODO: This has a strong implicit assumption of 1 spectrum per spectrumlist.
    #       wwill need to track number of objects per data_product and hdu so that we
    #       yield back the right things in the right order.
    from time import time

    ts_diffs = []
    for (z, k, data_product_id, hdu) in map(lambda _: map(int, _.split("_")), input_names):

        t_a = time()
        key = (data_product_id, hdu)
        spectrum = spectra[key][k]

        # TODO: check for failure.
        result = dict(zip(parameter_names, params[z]))
        result.update(dict(zip([f"e_{pn}" for pn in parameter_names], param_errs[z])))
        result.update(dict(zip([f"bitmask_{pn}" for pn in parameter_names], param_bitmask_flags[z])))
        result.update(
            log_chisq_fit=log_chisq_fit[z],
            log_snr_sq=log_snr_sq[z], 
            frac_phot_data_points=frac_phot_data_points[z],
        )
        t_b = time()
        try:
            ferre_time_elapsed = ferre_times_elapsed[z]
        except:
            ferre_time_elapsed = None
        
        t_c = time()

        # Database writes are the bottleneck. See if we can just yield and use the data product id and HDU.
        # TODO NOTE this will fail for individual visits.
        print(f"Using new output path")
        output_path = os.path.join(pwd, f"{data_product_id}_{hdu}.pkl")

        #output_path = os.path.join(pwd, f"{output.task.id}.pkl")
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #with open(output_path, "wb") as fp:
        ##   pickle.dump((model_flux[z], continuum[z]), fp, -1)

        print(f"Not creating data product, get it from pwd and output task id.")

        #output_data_product, _ = DataProduct.get_or_create(
        #    release="sdss5",
        #    filetype="full",
        #    kwargs=dict(full=output_path)
        #)
        #output.output_data_product = output_data_product        
        print(f"yielding {key} to ")
        t_d = time()
        g =  model(
            data_product=data_product_dict[data_product_id],
            spectrum=spectrum,
            ferre_n_obj=ferre_n_obj,
            ferre_time_load=ferre_time_load,
            ferre_time_elapsed=ferre_time_elapsed,
            ferre_timeout=ferre_timeout,
            **result
        )
        t_e = time()
        yield g
        t_f = time()

        ts = np.array([t_a, t_b, t_c, t_d, t_e, t_f])
        ts_diffs.append(ts)
        #print(ts)
        #print(100 * np.diff(ts)/np.ptp(ts))

    #for key, output in outputs.items():
    #    print(f"yielding {key} and {output}")
    #    yield output

    ts_diffs = np.array(ts_diffs)
    foo = np.sum(np.diff(ts_diffs, axis=1), axis=0)
    print(f"SUMMARY")
    print(foo / np.sum(foo))
