using HDF5
include("Grok.jl"); using .Grok # perhaps to split into a separate package at some point

# run: julia rungrok.jl mwmStar_paths.txt output.h5
infile = ARGS[1] # each line is an HDU index and a path to a mwmStar file, spearated by a space
outfile = ARGS[2] # output file (hdf5)

# grab the HDU indices and paths to the mwmStar files
const hdus, paths = let
    x = split.(readlines(infile))
    parse.(Int, first.(x)), last.(x)
end

# arrays to hold the results for each spectrum
# fill with NaNs initially, so that which stars were skipped is clear
const Nspec = length(paths)
const Nwl = length(Grok.ferre_wls)
const best_nodes = zeros(length(Grok.all_vals), Nspec) .* NaN 
const stellar_params = zeros(length(Grok.all_vals) + 1, Nspec) .* NaN
const detailed_abundances = zeros(length(Grok.elements_to_fit), Nspec) .* NaN
const best_fit_spectra = zeros(Nwl, Nspec) .* NaN
const obs_spectra = zeros(Nwl, Nspec) .* NaN
const obs_ivars = zeros(Nwl, Nspec) .* NaN
const pixel_masks = zeros(Nwl, Nspec) .* NaN
const runtimes = zeros(Nspec)

# process all the files 
for (i, (file, hdu_index)) in enumerate(zip(paths, hdus))
    println("processing HDU $hdu_index of $file")
    hdu_index += 1 # account for Julia's 1-based indexing

    try
        data = Grok.read_mwmStar(file, hdu_index)
        # stick the observed spectrum in the output arrays
        obs_spectra[:, i], obs_ivars[:, i], pixel_masks[:, i] = data
        t = @elapsed begin 
            # do the analysis and stick the results in the output arrays
            best_nodes[:, i], stellar_params[:, i], detailed_abundances[:, i], best_fit_spectra[:, i] = 
                Grok.analyse_spectrum(data...)
        end
        runtimes[i] = t
    catch e
        #println("encountered an error: $(typeof(e))")
        rethrow(e)
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
