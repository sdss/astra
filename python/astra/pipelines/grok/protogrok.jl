using HDF5, Optim, FITSIO, Interpolations, Korg
using DSP: conv

infile = ARGS[1]
outfile = ARGS[2]

include("element_windows.jl")

# first cli argument is a file containing paths of the mwmStar files to process
const hdus, paths = let
    x = split.(readlines(infile))
    parse.(Int, first.(x)), last.(x)
end

# read the mask that is aplied to all spectra
const global_ferre_mask = parse.(Bool, readlines(ENV["MWM_ASTRA"] * "/pipelines/Grok/ferre_mask.dat"));

# load grid of precomputed synthetic spectra
const gridfile = ENV["MWM_ASTRA"] * "/pipelines/Grok/korg_grid.h5" 
const all_vals = [h5read(gridfile, "Teff_vals"),
                  h5read(gridfile, "logg_vals"),
                  h5read(gridfile, "vmic_vals"),
                  h5read(gridfile, "metallicity_vals")]
const labels = ["Teff", "logg", "vmic", "metallicity"]
const model_spectra = let 
    spectra = h5read(gridfile, "spectra")
    # this corrected mask fixes my off-by-one error in the grid generation
    corrected_mask = [global_ferre_mask[2:end] ; false]
    spectra[corrected_mask, :, :,  :, :]
end
model_spectra[isnan.(model_spectra)] .= Inf; # NaNs (failed syntheses) will mess up the grid search.
const ferre_wls = (10 .^ (4.179 .+ 6e-6 * (0:8574)))[global_ferre_mask]

# set up a spectrum interpolator
const spectrum_itp = let xs = (1:sum(global_ferre_mask), all_vals...)
    linear_interpolation(xs, model_spectra)
end

# setup for live synthesis
const synth_wls = ferre_wls[1] - 10 : 0.01 : ferre_wls[end] + 10
const LSF = Korg.compute_LSF_matrix(synth_wls, ferre_wls, 22_500)
const linelist = Korg.get_APOGEE_DR17_linelist();
const elements_to_fit = ["Mg", "Na", "Al"] # these are what will be fit

function apply_rotation(flux, vsini; ε=0.6, log_lambda_step=1.3815510557964276e-5)
    Δlnλ_max = vsini * 1e5 / Korg.c_cgs
    p = Δlnλ_max / log_lambda_step
    Δlnλs = [-Δlnλ_max ; (-floor(p) : floor(p))*log_lambda_step ; Δlnλ_max]
    
    c1 = 2(1-ε)
    c2 = π * ε / 2
    c3 = π * (1-ε/3)
    
    x = @. (1 - (Δlnλs/Δlnλ_max)^2)
    rotation_kernel = @. (c1*sqrt(x) + c2*x)/(c3 * Δlnλ_max)
    rotation_kernel ./= sum(rotation_kernel)
    offset = (length(rotation_kernel) - 1) ÷ 2

    conv(rotation_kernel, flux)[offset+1 : end-offset]
end

function analyse_spectrum(flux, ivar, pixel_mask)
    flux = view(flux, pixel_mask)
    ivar = view(ivar, pixel_mask)

    # find closed node in the grid of synthetic spectra
    chi2 = sum((view(model_spectra, pixel_mask, :, :, :, :) .- flux).^2 .* ivar, dims=1)
    best_inds = collect(Tuple(argmin(chi2)))[2:end]
    best_node = getindex.(all_vals, best_inds)
    println("best node: ", best_node)

    # get stellar params via linear interpolation of grid
    lower_bounds = [first.(all_vals) ; 0.0]
    upper_bounds = [last.(all_vals) ; 500.0]
    p0 = [best_node ; 25.0]
    itp_res = optimize(lower_bounds, upper_bounds, p0) do p
        if any(.! (lower_bounds .<= p .<= upper_bounds))
            #@info "cost function called on OOB params $p"
            return 1.0 * length(flux) # nice high chi2
        end
        interpolated_spectrum = spectrum_itp(1:sum(global_ferre_mask), p[1:4]...)[pixel_mask]
        model_spectrum = apply_rotation(interpolated_spectrum, p[5])
        sum(@. (flux - model_spectrum)^2 .* ivar)
    end

    println("stellar params: ", itp_res.minimizer)

    #fixed_params = Dict(["Teff", "logg", "vmic", "m_H"] .=> itp_res.minimizer)
    #detailed_abundances = map(elements_to_fit) do el
    #    try
    #        Korg.Fit.fit_spectrum(ferre_wls[pixel_mask], flux, ivar.^(-1/2), linelist,
    #                              Dict(el=>itp_res.minimizer[end]), fixed_params;
    #                              synthesis_wls=synth_wls, LSF_matrix=LSF[pixel_mask, :],
    #                              windows=element_windows[el]).best_fit_params[el]
    #    catch e
    #        NaN
    #    end
    #end

    ## synthesize a final spectrum
    #synth_abunds = copy(detailed_abundances) # if an abundance failed, use m_H
    #synth_abunds[isnan.(synth_abunds)] .= fixed_params["m_H"] 
    #A_X = format_A_X(fixed_params["m_H"], Dict(elements_to_fit .=> synth_abunds))
    #atm = interpolate_marcs(fixed_params["Teff"], fixed_params["logg"], A_X; clamp_abundances=true)
    #sol = synthesize(atm, linelist, A_X, [synth_wls])
    #model_flux = LSF * (sol.flux ./ sol.cntm)

    best_node, itp_res.minimizer, zeros(length(elements_to_fit)), zeros(length(ferre_wls))#detailed_abundances, model_flux
end

function read_mwmStar(file, hdu_index; minimum_flux_sigma=0.05, bad_pixel_mask=16639)
     FITS(file) do f
        cntm = read(f[hdu_index], "continuum")[global_ferre_mask]
        flux = read(f[hdu_index], "flux")[global_ferre_mask] ./ cntm
        ivar = read(f[hdu_index], "ivar")[global_ferre_mask] .* cntm.^2

        # set minimum pixel error to minimum_flux_sigma
        ivar .= min.(ivar, minimum_flux_sigma^(-2))

        pixel_flags = Int.(read(f[hdu_index], "pixel_flags")[global_ferre_mask])
        pixel_mask = (pixel_flags .& bad_pixel_mask) .== 0
        pixel_mask .&= isfinite.(flux) .& isfinite.(ivar)
        pixel_mask .&= (flux .>= 0) .| (ivar .>= 0)

        flux, ivar, pixel_mask
    end
end

# arrays to hold the results for each spectrum
# fill with NaNs initially, so that which stars were skipped is clear
Nspec = length(paths)
best_nodes = zeros(4, Nspec) .* NaN 
stellar_params = zeros(length(all_vals) + 1, Nspec) .* NaN
detailed_abundances = zeros(length(elements_to_fit), Nspec) .* NaN
best_fit_spectra = zeros(length(ferre_wls), Nspec) .* NaN
obs_spectra = zeros(length(ferre_wls), Nspec) .* NaN
obs_ivars = zeros(length(ferre_wls), Nspec) .* NaN
pixel_masks = zeros(length(ferre_wls), Nspec) .* NaN
runtimes = zeros(Nspec)

# process all the files 
for (i, (file, hdu_index)) in enumerate(zip(paths, hdus))
    println("processing HDU $hdu_index of $file")
    hdu_index += 1 # account for Julia's 1-based indexing

    try
        data = read_mwmStar(file, hdu_index)
        # stick the observed spectrum in the output arrays
        obs_spectra[:, i], obs_ivars[:, i], pixel_masks[:, i] = data
        t = @elapsed begin 
            # do the analysis and stick the results in the output arrays
            best_nodes[:, i], stellar_params[:, i], detailed_abundances[:, i], best_fit_spectra[:, i] = 
                analyse_spectrum(data...)
        end
        runtimes[i] = t
    catch e
        println("encountered an error: $(typeof(e))")
        #rethrow(e)
    end
end

# save everything in an HDF5 file
h5open(outfile, "w") do f
    f["best_nodes"] = best_nodes
    f["stellar_params"] = stellar_params
    f["detailed_abundances"] = detailed_abundances
    f["model_spectra"] = best_fit_spectra
    f["runtimes"] = runtimes
    f["obs_spectra"] = obs_spectra
    f["obs_ivars"] = obs_ivars
    f["pixel_masks"] = pixel_masks
end
