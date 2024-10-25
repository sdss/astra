using Pkg

Pkg.status("Korg")

using Korg


function synthesize(atmosphere_path, linelist_path, metallicity)
    println("Running with ", atmosphere_path, " and ", linelist_path)
    println("Timing atmosphere read")
    @time atm = Korg.read_model_atmosphere(atmosphere_path)
    println("Timing line list read")
    @time linelist = Korg.read_linelist(linelist_path, format="{korg_read_transitions_format}")
    A_X = Korg.format_A_X(metallicity, {solar_abundances_formatted}; solar_relative={solar_relative})
    println("Timing synthesis")
    @time spectrum = Korg.synthesize(atm, linelist, A_X, {lambda_vacuum_min}, {lambda_vacuum_max}; hydrogen_lines={hydrogen_lines}, vmic={microturbulence:.2f})
    println("Done")
    return spectrum
end

println("Time 1 start")
@time spectrum = synthesize("{atmosphere_path}", "{linelist_path}", {fake_metallicity:.2f})
println("Time 1 end")

println("Time 2 start")
@time spectrum = synthesize("{atmosphere_path}", "{linelist_path}", {metallicity:.2f})
println("Time 2 end")


# Save to disk.
open("spectrum.out", "w") do fp
    for (wl, flux) in zip(spectrum.wavelengths, spectrum.flux)
        println(fp, wl, " ", flux)
    end
end

# Now do continuum.
atm = Korg.read_model_atmosphere("{atmosphere_path}")
A_X = Korg.format_A_X({metallicity:.2f}, {solar_abundances_formatted}; solar_relative={solar_relative})
continuum = Korg.synthesize(atm, [], A_X, {lambda_vacuum_min}, {lambda_vacuum_max}; hydrogen_lines=false, vmic={microturbulence:.2f})
open("continuum.out", "w") do fp
    for (wl, flux) in zip(continuum.wavelengths, continuum.flux)
        println(fp, wl, " ", flux)
    end
end
