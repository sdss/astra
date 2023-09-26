using Pkg

Pkg.status("Korg")

using Korg

function read_line_list(linelist_path_0, linelist_path_1)
    linelist_1 = Korg.read_linelist(linelist_path_0, format="{korg_read_transitions_format}")
    linelist_2 = Korg.read_linelist(linelist_path_1, format="{korg_read_transitions_format}")
    linelist = append!(linelist_1, linelist_2)
    return linelist
end

function synthesize(atmosphere_path, linelist_path_0, linelist_path_1, metallicity)
    println("Running with ", atmosphere_path, " and ", linelist_path_0, " and ", linelist_path_1)
    println("Timing atmosphere read")
    @time atm = Korg.read_model_atmosphere(atmosphere_path)
    println("Timing line list read")
    @time linelist = read_line_list(linelist_path_0, linelist_path_1)
    println("Timing synthesis")
    @time spectrum = Korg.synthesize(atm, linelist, {lambda_vacuum_min:.2f}, {lambda_vacuum_max:.2f}; metallicity=metallicity, hydrogen_lines={hydrogen_lines}, vmic={microturbulence:.2f}, abundances={abundances_formatted}, solar_relative={solar_relative})
    println("Done")
    return spectrum
end


println("Time 1 start")
@time spectrum = synthesize("{atmosphere_path}", "{linelist_path_0}", "{linelist_path_1}", {fake_metallicity:.2f})
println("Time 1 end")

println("Time 2 start")
@time spectrum = synthesize("{atmosphere_path}", "{linelist_path_0}", "{linelist_path_1}", {metallicity:.2f})
println("Time 2 end")

# Save to disk.
open("spectrum.out", "w") do fp
    for (wl, flux) in zip(spectrum.wavelengths, spectrum.flux)
        println(fp, wl, " ", flux)
    end
end
println("Sold!")

# Now do continuum.
atm = Korg.read_model_atmosphere("{atmosphere_path}")
continuum = Korg.synthesize(atm, [], {lambda_vacuum_min:.2f}, {lambda_vacuum_max:.2f}; metallicity={metallicity:.2f}, hydrogen_lines=false, vmic={microturbulence:.2f}, abundances={abundances_formatted}, solar_relative={solar_relative})
open("continuum.out", "w") do fp
    for (wl, flux) in zip(continuum.wavelengths, continuum.flux)
        println(fp, wl, " ", flux)
    end
end
