using Pkg

Pkg.status("Korg")

using Korg


#usage example
#apolines = [parse_ts_linelist("../linelists/APOGEE/20180901/turbospec.20180901.atoms");
#            parse_ts_linelist("../linelists/APOGEE/20180901/turbospec.20180901.molec"; mols=true)]
#
#sort!(apolines, by=l->l.wl)

function parse_ts_linelist(fn; mols=false)
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
            isotopic_correction = log10(Korg.isotopic_abundances[(el1, m1)] * Korg.isotopic_abundances[(el2, m2)])
        
            transitions = Korg.parse_fwf(lines[lb:ub], [
                       (2:10, Float64, :wl, x->Korg.air_to_vacuum(x)*1e-8),
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
                log10(Korg.isotopic_abundances[(Korg.get_atoms(species.formula)[1], m)])
            end

            transitions = Korg.parse_fwf(lines[lb:ub], [
                       (2:10, Float64, :wl, x->Korg.air_to_vacuum(x)*1e-8),
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

#usage example
#apolines = [parse_ts_linelist("../linelists/APOGEE/20180901/turbospec.20180901.atoms");
#            parse_ts_linelist("../linelists/APOGEE/20180901/turbospec.20180901.molec"; mols=true)]
#
#sort!(apolines, by=l->l.wl)

function read_line_list()

    linelist = [parse_ts_linelist("{linelist_path_0}");
                parse_ts_linelist("{linelist_path_1}"; mols=true)]
    sort!(linelist, by=l->l.wl)
    return linelist
end


function synthesize(atmosphere_path, metallicity)
    println("Running with ", atmosphere_path)
    println("Timing atmosphere read")
    @time atm = Korg.read_model_atmosphere(atmosphere_path)
    println("Timing line list read")
    @time linelist = read_line_list()
    println("Timing synthesis")
    @time spectrum = Korg.synthesize(atm, linelist, {lambda_vacuum_min:.2f}, {lambda_vacuum_max:.2f}; metallicity=metallicity, hydrogen_lines={hydrogen_lines}, vmic={microturbulence:.2f}, abundances={abundances_formatted}, solar_relative={solar_relative})
    return spectrum
end


println("Time 1 start")
@time spectrum = synthesize("{atmosphere_path}", {fake_metallicity:.2f})
println("Time 1 end")

println("Time 2 start")
@time spectrum = synthesize("{atmosphere_path}", {metallicity:.2f})
println("Time 2 end")

# Save to disk.
open("spectrum.out", "w") do fp
    for (wl, flux) in zip(spectrum.wavelengths, spectrum.flux)
        println(fp, wl, " ", flux)
    end
end

# Now do continuum.
atm = Korg.read_model_atmosphere("{atmosphere_path}")
continuum = Korg.synthesize(atm, [], {lambda_vacuum_min:.2f}, {lambda_vacuum_max:.2f}; metallicity={metallicity:.2f}, hydrogen_lines=false, vmic={microturbulence:.2f}, abundances={abundances_formatted}, solar_relative={solar_relative})
open("continuum.out", "w") do fp
    for (wl, flux) in zip(continuum.wavelengths, continuum.flux)
        println(fp, wl, " ", flux)
    end
end
