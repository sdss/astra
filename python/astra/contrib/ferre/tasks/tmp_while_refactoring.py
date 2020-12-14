

class InitialEstimateOfStellarParametersGivenApStarFileBase(WrapperTask, SDSS5ApStarMixin, FerreMixin):


    
    analysis_task = FerreGivenApStarFile
    

    def dispatcher(self):

        with open(self.grid_header_list_path, "r") as fp:
            grid_header_paths = list(map(str.strip, fp.readlines()))
        
        common_kwds = self.get_common_param_kwargs(self.analysis_task)
        grid_limits = utils.parse_grid_limits(grid_header_paths)

        sdss_paths = {}
        for i, batch_kwds in enumerate(self.get_batch_task_kwds(include_non_batch_keywords=False)):
            try:
                sdss_path = sdss_paths[self.release]
            except KeyError:
                sdss_paths[self.release] = sdss_path = SDSSPath(
                    release=self.release,
                    public=self.public,
                    mirror=self.mirror,
                    verbose=self.verbose
                )

            path = sdss_path.full("apStar", **batch_kwds)

            try:
                header = getheader(path)
                teff = utils.safe_read_header(header, ("RV_TEFF", "RVTEFF"))
                logg = utils.safe_read_header(header, ("RV_LOGG", "RVLOGG"))
                fe_h = utils.safe_read_header(header, ("RV_FEH", "RVFEH"))
                kwds = {
                    "mean_fiber": header["MEANFIB"],
                    "teff": teff,
                    "logg": logg,
                    "fe_h": fe_h
                }

            except:
                log.exception("Exception: ")
                continue

            else:
                batch_kwds.update(
                    initial_parameters={
                        "TEFF": teff,
                        "LOGG": logg,
                        "METALS": fe_h,
                        "LOG10VDOP": utils.approximate_log10_microturbulence(logg),
                        "O Mg Si S Ca Ti": 0,
                        "LGVSINI": 0,
                        "C": 0,
                        "N": 0,
                    }
                )

                any_suitable_grids = False
                for grid_header_path, parsed_header_path in utils.yield_suitable_grids(grid_limits, **kwds):
                    any_suitable_grids = True
                    
                    # In the initial FERRE run we freeze LOG10VDOP.
                    yield_kwds = dict(
                        grid_header_path=grid_header_path,
                        frozen_parameters=dict(LOG10VDOP=None),
                    )
                    # Freeze C and N to zero if this is a dwarf grid.
                    if parsed_header_path["gd"] == "d":
                        yield_kwds["frozen_parameters"].update(C=0.0, N=0.0)

                    yield_kwds.update({ **common_kwds, **batch_kwds })
                    
                    # We yield an integer so we can see progress of unique objects.
                    yield (i, yield_kwds)
                    

    def requires(self):
        with tqdm(desc="Dispatching", total=self.get_batch_size()) as pbar:
            for iteration, kwds in self.dispatcher():
                pbar.update(iteration)
                yield self.analysis_task(**kwds)
                

class BestInitialEstimateOfStellarParametersGivenApStarFile(SDSS5ApStarMixin, FerreMixin):

    def requires(self):
        """ Requirements of this task. """
        return InitialEstimateOfStellarParametersGivenApStarFileBase(
            **self.get_common_param_kwargs(InitialEstimateOfStellarParametersGivenApStarFileBase)
        )


    def uid(self, task):
        batch_param_names = self.requires().analysis_task.observation_task.batch_param_names()
        return "_".join([f"{getattr(task, pn)}" for pn in batch_param_names])


    def get_best_initial_estimates(self):
        """ Gest best initial estimate. """
        
        best_tasks = {}
        for task in self.requires().requires():
            key = self.uid(task)
            best_tasks.setdefault(key, (np.inf, None))

            output = task.output()["database"].read(as_dict=True)

            log_chisq_fit = output["log_chisq_fit"]

            parsed_header = utils.parse_header_path(task.grid_header_path)
            
            # Penalise chi-sq in the same way they did for DR16.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
            if parsed_header["spectral_type"] == "GK" and output["TEFF"] < 3985:
                # \chi^2 *= 10
                log_chisq_fit += np.log(10)

            if log_chisq_fit < best_tasks[key][0]:
                best_tasks[key] = (log_chisq_fit, task, output)

        return best_tasks



    def run(self):

        best_initial_estimates = self.get_best_initial_estimates()

        for task in self.get_batch_tasks():
            try:
                log_chisq_fit, initial_task, output = best_initial_estimates[self.uid(task)]

            except KeyError:
                log.exception(f"No initial guess for {task}")
                kwds = {}

            else:
                kwds = dict(
                    grid_header_path=initial_task.grid_header_path,
                    initial_parameters={ k: v for k, v in output.items() if k == k.upper() }
                )

            finally:
                with task.output().open("w") as fp:
                    fp.write(json.dumps(kwds))
                                

    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return MockTarget(self.task_id)





class CreateMedianFilteredApStarFile(SDSS5ApStarMixin, FerreMixin):


    median_filter_width = astra.IntParameter(default=151)
    bad_minimum_flux = astra.FloatParameter(default=0.01)
    non_finite_err_value = astra.FloatParameter(default=1e10)

    def requires(self):
        return InitialEstimateOfStellarParametersGivenApStarFileBase(
            **self.get_common_param_kwargs(InitialEstimateOfStellarParametersGivenApStarFileBase)
        )


    def uid(self, task):
        batch_param_names = self.requires().analysis_task.observation_task.batch_param_names()
        return "_".join([f"{getattr(task, pn)}" for pn in batch_param_names])


    def get_best_initial_estimates(self):
        """ Gest best initial estimate. """
        
        best_tasks = {}
        for task in self.requires().requires():
            key = self.uid(task)
            best_tasks.setdefault(key, (np.inf, None))

            output = task.output()["database"].read(as_dict=True)

            log_chisq_fit = output["log_chisq_fit"]

            parsed_header = utils.parse_header_path(task.grid_header_path)
            
            # Penalise chi-sq in the same way they did for DR16.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
            if parsed_header["spectral_type"] == "GK" and output["TEFF"] < 3985:
                # \chi^2 *= 10
                log_chisq_fit += np.log(10)

            if log_chisq_fit < best_tasks[key][0]:
                best_tasks[key] = (log_chisq_fit, task, output)

        return best_tasks


    def run(self):

        best_initial_estimates = self.get_best_initial_estimates()

        # Get best estimate per task.
        for task in self.get_batch_tasks():
            try:
                log_chisq_fit, initial_task, output = best_initial_estimates[self.uid(task)]

            except KeyError:
                # Create an empty file?
                # TODO: How can we change this...?
                with open(task.output().path, "w") as fp:
                    fp.write("")
                continue

            # Re-normalize the spectrum using the previous estimate.
            image = fits.open(initial_task.output()["AstraSource"].path)

            # Get segments for each chip based on the model.
            n_pixels = [header["NPIX"] for header in utils.read_ferre_headers(initial_task.grid_header_path)][1:]

            with open(initial_task.input_wavelength_mask_path, "rb") as fp:
                mask = pickle.load(fp)

            indices = 1 + np.cumsum(mask).searchsorted(np.cumsum(n_pixels))
            # These indices will be for each chip, but will need to be left-trimmed.
            segment_indices = np.sort(np.hstack([
                0,
                np.repeat(indices[:-1], 2),
                mask.size
            ])).reshape((-1, 2))
            
            # Left-trim the indices.
            for i, (start, end) in enumerate(segment_indices):
                segment_indices[i, 0] += mask[start:].searchsorted(True)
            
            continuum = median_filtered_correction(
                wavelength=np.arange(image[1].data[0].size),
                normalised_observed_flux=image[1].data[0],
                normalised_observed_flux_err=image[2].data[0]**-0.5,
                normalised_model_flux=image[5].data[0],
                segment_indices=segment_indices,
                width=self.median_filter_width,
                bad_minimum_flux=self.bad_minimum_flux,
                non_finite_err_value=self.non_finite_err_value                
            )

            '''
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12,3))

            for i, moo in enumerate(segment_indices):
                ax.plot(
                    observation.wavelength.value[moo[0]:moo[1]],
                    image[1].data[0][moo[0]:moo[1]],
                    c="tab:blue"
                )
                ax.plCreateMedianFilteredApStarFileot(
                    observation.wavelength.value[moo[0]:moo[1]],
                    image[1].data[0][moo[0]:moo[1]] / continuum[moo[0]:moo[1]],
                    c="k"
                )

                ax.plot(
                    observation.wavelength.value[moo[0]:moo[1]],
                    0.1 + continuum[moo[0]:moo[1]],
                    c="tab:red"
                )
            fig.savefig("tmp.png")

            '''

            # Copy the original file to the output file, then change the flux.
            new_image = fits.open(initial_task.input()["observation"].path)
            new_image[1].data /= continuum
            new_image[2].data /= continuum
            new_image.writeto(task.output().path, overwrite=True)
            
        
    
    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        # TODO: To be defined by SDSS5/SDSS4 mixin
        new_path = AstraSource(self).path.replace("/AstraSource", "/ApStar")
        return LocalTarget(new_path)


    

class RevisedEstimateOfStellarParametersGivenApStarFile(SDSS5ApStarMixin, FerreMixin):

    def requires(self):
        return {
            "observation": CreateMedianFilteredApStarFile(**self.get_common_param_kwargs(CreateMedianFilteredApStarFile)),
            "initial_estimate": BestInitialEstimateOfStellarParametersGivenApStarFile(**self.get_common_param_kwargs(BestInitialEstimateOfStellarParametersGivenApStarFile))
        }

    
    def run(self):

        task_factory = next(self.get_batch_tasks()).requires()["initial_estimate"].requires().analysis_task

        batch_tasks = []
        execute_tasks = []

        for i, task in enumerate(self.get_batch_tasks()):
            # Check initial estimate.
            with task.input()["initial_estimate"].open("r") as fp:
                additional_kwds = json.load(fp)

            if not additional_kwds: 
                # No initial estimate. Outside of grid bounds.
                task.output()["database"].write()
                continue

            kwds = task.get_common_param_kwargs(task_factory)
            kwds.update(additional_kwds)

            batch_tasks.append(task)
            execute_tasks.append(task_factory(**kwds))
            
        outputs = yield execute_tasks

        for task, output in zip(batch_tasks, outputs):
            # Write database output.
            task.output()["database"].write(
                output["database"].read(as_dict=True, include_parameters=False)
            )

            # Copy AstraSource object.




    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return {
            "database": FerreResult(self),
            #"AstraSource": AstraSource(self)
        }


class EstimateAbundanceGivenApStarFile(SDSS5ApStarMixin, FerreMixin):

    def requires(self):
        return {
            "observation": CreateMedianFilteredApStarFile(**self.get_common_param_kwargs(CreateMedianFilteredApStarFile)),
            "stellar_parameters": RevisedEstimateOfStellarParametersGivenApStarFile(
                **self.get_common_param_kwargs(RevisedEstimateOfStellarParametersGivenApStarFile)
            )
        }


    def run(self):

        for i, task in enumerate(self.get_batch_tasks()):
            if i > 0:

                raise a
        

    def output(self):
        return AstraSource(self)