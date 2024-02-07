module Grok
using HDF5, Optim, FITSIO, Interpolations, Korg
using DSP: conv

include("element_windows.jl")

# read the mask that is aplied to all spectra
const global_ferre_mask = parse.(Bool, readlines("ferre_mask.dat"));

# load grid of precomputed synthetic spectra
gridfile = ENV["GROK_GRID_FILE"]
const all_vals = [h5read(gridfile, "Teff_vals"),
                  h5read(gridfile, "logg_vals"),
                  #h5read(gridfile, "vmic_vals"),
                  h5read(gridfile, "metallicity_vals")]
const vmic_index = 5 # 1 km/s        

const labels = ["Teff", "logg", "metallicity"] 
const model_spectra = let 
    spectra = h5read(gridfile, "spectra")
    # this corrected mask fixes my off-by-one error in the grid generation
    corrected_mask = [global_ferre_mask[2:end] ; false]
    spectra[corrected_mask, :, :, vmic_index, :]
end
model_spectra[isnan.(model_spectra)] .= Inf; # NaNs (failed syntheses) will mess up the grid search.
const ferre_wls = (10 .^ (4.179 .+ 6e-6 * (0:8574)))[global_ferre_mask]

"""
Convert an array to a range if it is possible to do so.
"""
function rangify(a::AbstractRange)
    a
end
function rangify(a::AbstractVector)
    minstep, maxstep = extrema(diff(a))
    @assert minstep ≈ maxstep
    range(a[1], a[end]; length=length(a))
end

# set up a spectrum interpolator
interpolate_spectrum = let 
    T_pivot = 3000 # TODO change this to 4000 when the grid is updated
    coolmask = all_vals[1] .< T_pivot

    # uncomment this to use grids that go below the temperature pivot
    #xs = rangify.((1:sum(global_ferre_mask), all_vals[1][coolmask], all_vals[2:end]...))
    #cool_itp = cubic_spline_interpolation(xs, model_spectra[:, coolmask, :, :, :])

    xs = rangify.((1:sum(global_ferre_mask), all_vals[1][.! coolmask], all_vals[2:end]...))
    hot_itp = cubic_spline_interpolation(xs, model_spectra[:, .!coolmask, :, :]) 
    
    function itp(Teff, logg, metallicity) 
        if Teff < T_pivot
            #cool_itp(1:sum(global_ferre_mask), Teff, logg, vmic, metallicity)
        else
            hot_itp(1:sum(global_ferre_mask), Teff, logg, metallicity)
        end
    end
end

# setup for live synthesis
const synth_wls = ferre_wls[1] - 10 : 0.01 : ferre_wls[end] + 10
const LSF = Korg.compute_LSF_matrix(synth_wls, ferre_wls, 22_500)
const linelist = Korg.get_APOGEE_DR17_linelist();
const elements_to_fit = ["Mg", "Na", "Al"] # these are what will be fit

"""
We don't use Korg's `apply_rotation` because this is specialed for the case where wavelenths are 
log-uniform.  We can use FFT-based convolution to apply the rotation kernel to the spectrum in this 
case.
"""
function apply_rotation(flux, vsini; ε=0.6, log_lambda_step=1.3815510557964276e-5)
    # half-width of the rotation kernel in Δlnλ
    Δlnλ_max = vsini * 1e5 / Korg.c_cgs
    # half-width of the rotation kernel in pixels
    p = Δlnλ_max / log_lambda_step
    # Δlnλ detuning for each pixel in kernel
    Δlnλs = [-Δlnλ_max ; (-floor(p) : floor(p))*log_lambda_step ; Δlnλ_max]
    
    # kernel coefficients
    c1 = 2(1-ε)
    c2 = π * ε / 2
    c3 = π * (1-ε/3)
    
    x = @. (1 - (Δlnλs/Δlnλ_max)^2)
    rotation_kernel = @. (c1*sqrt(x) + c2*x)/(c3 * Δlnλ_max)
    rotation_kernel ./= sum(rotation_kernel)
    offset = (length(rotation_kernel) - 1) ÷ 2

    conv(rotation_kernel, flux)[offset+1 : end-offset]
end

"""
Given an observed spectrum, compute
- the closest spectrum in the grid of synthetic spectra
- the best-fit stellar parameters via linear interpolation of the grid
- best-fit params and abundances from live synthesis (currently dissabled)
"""
function analyse_spectrum(flux, ivar, pixel_mask)
    flux = view(flux, pixel_mask)
    ivar = view(ivar, pixel_mask)

    # find closed node in the grid of synthetic spectra
    chi2 = sum((view(model_spectra, pixel_mask, :, :, :) .- flux).^2 .* ivar, dims=1)
    best_inds = collect(Tuple(argmin(chi2)))[2:end]
    best_node = getindex.(all_vals, best_inds)
    println("best node: ", best_node)

    # get stellar params via linear interpolation of grid
    lower_bounds = [first.(all_vals) ; 0.0]
    upper_bounds = [last.(all_vals) ; 500.0]
    p0 = [best_node ; 25.0]

    #itp_res = optimize(lower_bounds, upper_bounds, p0) do p
    #    if any(.! (lower_bounds .<= p .<= upper_bounds))
    #        @info "cost function called on OOB params $p"
    #        return 1.0 * length(flux) # nice high chi2
    #    end
    #    interpolated_spectrum = interpolate_spectrum(p[1:3]...)[pixel_mask]
    #    model_spectrum = apply_rotation(interpolated_spectrum, p[4])
    #    sum(@. (flux - model_spectrum)^2 .* ivar)
    #end

    function f(p)
        if any(.! (lower_bounds .<= p .<= upper_bounds))
            #@info "cost function called on OOB params $p"
            return 1.0 * length(flux) # nice high chi2
        end
        #println(p)
        interpolated_spectrum = interpolate_spectrum(p[1:3]...)[pixel_mask]
        model_spectrum = apply_rotation(interpolated_spectrum, p[4])
        sum(@. (flux - model_spectrum)^2 .* ivar)
    end

    #itp_res = optimize(f, p0, NelderMead())
    itp_res = optimize(f, lower_bounds, upper_bounds, p0, NelderMead())

    println("stellar params: ", itp_res.minimizer)

    # de-mask it based on pixel mask
    model_flux = zeros(length(pixel_mask))
    if any(.! (lower_bounds .<= itp_res.minimizer .<= upper_bounds))
        @info "optimised value is out of bounds"
    else
        model_flux[pixel_mask] .= apply_rotation(interpolate_spectrum(itp_res.minimizer[1:3]...)[pixel_mask], itp_res.minimizer[4])
    end    

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

    best_node, itp_res.minimizer, zeros(length(elements_to_fit)), model_flux #detailed_abundances, model_flux
end

"""
Given a path and HDU index, pull a spectrum from an mwmStar file and preprocess it for analysis.
"""
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

end # module
