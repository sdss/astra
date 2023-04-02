import numpy as np
from astropy.table import Table
from astra.utils import expand_path
from sdss_access import SDSSPath


aspcap = Table.read(expand_path("$MWM_ASTRA/0.2.6/v6_0_9-1.0/results/summary/astraAllStar-ASPCAP-0.2.6-v6_0_9-1.0.fits"))

mask = (
    (aspcap["SNR"] > 200)
&   (aspcap["LOGG"] < 3.0)
&   (aspcap["TEFF"] < 6000)
&   np.isfinite(aspcap["TEFF"])
&   np.isfinite(aspcap["LOGG"])
&   np.isfinite(aspcap["METALS"])
&   np.isfinite(aspcap["O_MG_SI_S_CA_TI"])
&   np.isfinite(aspcap["C_H_PHOTOSPHERE"])
)

np.random.seed(0)

indices = np.random.choice(np.where(mask)[0], 5000, replace=False)

code = """
using Distributed
Distributed.addprocs(128)

@everywhere using Korg
@everywhere using FITSIO
@everywhere using ProgressMeter


@everywhere function parse_ts_linelist(atoms_fn, mols_fn)
    apolines = [parse_ts_linelist(atoms_fn);
                parse_ts_linelist(mols_fn; mols=true)]
    sort!(apolines, by=l->l.wl)
end

@everywhere function parse_ts_linelist(fn; mols=false)
    lines = readlines(fn)
    species_headers = findall(lines) do line
        line[1] == '\''
    end
    starts = species_headers[2:2:end]

    transitions = map(1:length(starts)) do i
        lb = starts[i] + 1
        ub = if i == length(starts)
            length(lines)
        else
            starts[i+1]-2
        end


        isotope_line = lines[starts[i]-1]
        species = join(split(strip(lines[lb-1][2:end-1]), ' ', keepempty=false), "_")
        if mols
            if species == "FEH"
                species = "FeH"
            end
            species = Korg.Species(species)
            
            isostring = split(isotope_line, '.')[2][1:6]
            m1 = parse(Int, isostring[1:3])
            m2 = parse(Int, isostring[4:6])
            el1, el2  = Korg.get_atoms(species.formula)
            if el2 == "H"
                el2 = el1
                el1 = "H" 
            end
            isotopic_correction = log10(Korg.isotopic_abundances[el1][m1] * Korg.isotopic_abundances[el2][m2])
        
            transitions = Korg.parse_fwf(lines[lb:ub], [
                       (2:10, Float64, :wl, x->Korg.air_to_vacuum(x*1e-8)),
                       (12:17, Float64, :Elo),
                       (19:25, Float64, :log_gf),
                       (41:48, Float64, :gamma_rad)
                    ])
            map(transitions) do transition
                gamma_rad = Korg.approximate_radiative_gamma(transition.wl, transition.log_gf)
                Korg.Line(transition.wl, transition.log_gf + isotopic_correction, species, 
                    transition.Elo, gamma_rad, 0.0, 0.0)
            end
        else
            elem, ion = split(species, '_')
            if length(elem) > 1
                elem = elem[1:1] * lowercase(elem[2:end])
            end
            species = Korg.Species(elem*"_"*ion)
            
            isostring = split(isotope_line, '.')[2][1:3]
            m = parse(Int, isostring)
            isotopic_correction = if m == 0
                0.0
            elseif m == 864 #weird barium case. Obviously this isn't a real isotope.  Don't correct logg 
                0.0
            else
                log10(Korg.isotopic_abundances[Korg.get_atoms(species.formula)[1]][m])
            end

            transitions = Korg.parse_fwf(lines[lb:ub], [
                       (2:10, Float64, :wl, x->Korg.air_to_vacuum(x*1e-8)),
                       (12:17, Float64, :Elo),
                       (19:25, Float64, :log_gf),
                       (31:35, Float64, :gamma_vdW),
                       (45:52, Float64, :gamma_rad)])
            map(transitions) do transition
                Korg.Line(transition.wl, transition.log_gf + isotopic_correction, species, 
                    transition.Elo, transition.gamma_rad, 0.0,  transition.gamma_vdW)
            end
        end
    end

    reduce(vcat, transitions)
end


@everywhere function best_fit_params(mwmStar_path, HDU_number, p0, apolines, synthesis_wls, obs_wls, LSF_matrix)
    obs_flux, obs_err = FITS(mwmStar_path) do f
        hdu = f[HDU_number+1] # adjust for julia's 1-based indexing
        read(hdu, "FLUX")[:], read(hdu, "E_FLUX")[:]
    end

    # sorry this is a mess. I'm currently returning a bunch of stuff for debugging's sake
    multires, _, _, _, small_obs_wls, _, _, small_obs_wl_range_inds = 
        Korg.Fit.find_best_params_multilocally(obs_wls, obs_flux, obs_err, apolines, p0, 
                                               synthesis_wls, LSF_matrix, verbose=false)
    windows = [(small_obs_wls[r[1]], small_obs_wls[r[end]]) for r in small_obs_wl_range_inds]

    if !multires.x_converged
        return [NaN, NaN, NaN, NaN, NaN], windows, [NaN, NaN, NaN, NaN, NaN]
    end

    multilocal_p_star = Korg.Fit.unscale(multires.minimizer)
    res = Korg.Fit.find_best_params_globally(obs_wls, obs_flux, obs_err, apolines, multilocal_p_star; 
                                             synthesis_wls=synthesis_wls, LSF_matrix=LSF_matrix, 
                                             verbose=false)
    p_star = if !res.x_converged
        [NaN, NaN, NaN, NaN, NaN]
    else
        Korg.Fit.unscale(res.minimizer)
    end

    multilocal_p_star, windows, p_star
end




# setup that should only happen once
@everywhere apolines = parse_ts_linelist(
    "/uufs/chpc.utah.edu/common/home/u6020307/sas_home/grok/data/transitions/turbospec.20180901.atoms", 
    "/uufs/chpc.utah.edu/common/home/u6020307/sas_home/grok/data/transitions/turbospec.20180901.molec"
)
@everywhere obs_wls = 10 .^ range(;start=log10(15100.802), step=6e-6, length=8575)
@everywhere synthesis_wls = obs_wls[1]-10:0.01:obs_wls[end]+10;
@everywhere LSF_matrix = Korg.Fit.compute_LSF_matrix(synthesis_wls, obs_wls, 22_500);

@everywhere function do_analysis(path, p0)
    println("starting ", path)
    @time multilocal_p_star, windows, p_star = best_fit_params(path, 3, p0, 
                                                            apolines, synthesis_wls, obs_wls, LSF_matrix)
    println(path)                                                    
    display(["p0" "multilocal" "global" ; p0 multilocal_p_star p_star])
    println("windows:")
    display(windows)
end
"""



p = SDSSPath("sdss5")
path, p0 = ([], [])

for index in indices:
    kwds = dict(
        v_astra="0.2.6",
        cat_id=aspcap["CAT_ID"][index],
        run2d="v6_0_9",
        apred="1.0"
    )
    mwmStar_path = p.full("mwmStar", **kwds)

    # Get initial teff/logg etc.
    keys = ("TEFF", "LOGG", "METALS", "O_MG_SI_S_CA_TI", "C_H_PHOTOSPHERE")
    p0.append([aspcap[key][index] for key in keys])
    path.append(mwmStar_path)



code += "paths = [\n"
for path_ in path:
    code += f'"{path_}",\n'
code += "]\n"

code += "p0s = [\n"
for p0_ in p0:
    code += f"{p0_},\n"
code += "]\n"


code += """
@showprogress pmap(x -> do_analysis(x...), zip(paths, p0s))
"""

with open("subset_script.jl", "w") as fp:
    fp.write(code)