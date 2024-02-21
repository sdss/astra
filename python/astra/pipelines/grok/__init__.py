import os
import numpy as np
from tqdm import tqdm
from typing import Iterable, Optional, Sequence
from peewee import JOIN, ModelSelect

from astra import task
from astra.models.grok import Grok
from astra.models.mwm import ApogeeCombinedSpectrum
from astra.utils import log, expand_path

@task
def grok(
    spectra: Optional[Iterable[ApogeeCombinedSpectrum]] = (
        ApogeeCombinedSpectrum
        .select()
        .join(Grok, JOIN.LEFT_OUTER, on=(ApogeeCombinedSpectrum.spectrum_pk == Grok.spectrum_pk))
        .where(Grok.spectrum_pk.is_null())
    ),
    grid_path: Optional[str] = "$MWM_ASTRA/pipelines/Grok/big_grid_2024-12-20.h5",
    mask_path: Optional[str] = "$MWM_ASTRA/pipelines/Grok/ferre_mask.dat",
    inflate_errors: Optional[bool] = True,
    use_subset: Optional[bool] = True,
    refinement_levels=[4, 4, 2],
    v_sinis: Optional[Sequence[float]] = [0, 5, 7.5, 10, 20, 30, 50, 75, 100],
    ε: Optional[float] = 0.6,
    fill_non_finite: Optional[float] = 10.0,
    use_median_ratio: Optional[bool] = False,
    use_nmf_flux: Optional[bool] = True,
    limit: Optional[int] = None,
    n_workers: Optional[int] = 0,
    **kwargs
) -> Iterable[Grok]:
    """
    Estimate the stellar parameters given a rest-frame resampled APOGEE spectrum.
    
    :param spectra:
        An iterable of APOGEE rest-frame spectra resampled onto the usual wavelength grid.
    
    :param grid_path: [optional]
        The path to the grid file.
    
    :param mask_path: [optional]
        The path to the mask file.
    
    :param inflate_errors: [optional]
        Inflate the flux uncertainties around significant skylines (or other spikes), and set bad
        pixels to have no useful data. This is equivalent to an ASPCAP pre-processing step.
    
    :param use_subset: [optional]
        Use only a subset of the grid. If given, the grid will be sliced to have [C/M] = [N/M] = 0 
        and v_micro = 1.0 km/s.

    :param refinement_levels: [optional]
        The number of refinement levels to use when searching for the best nodes in the grid.        

    :param v_sinis: [optional]
        Optionally provide a sequence of `v_sini` values (in units of km/s) to convolve the grid with.
    
    :param ε: [optional]
        The value of limb darkening to use when computing the `v_sini` kernels. This argument is
        ignored if `v_sinis` is None.
    
    :param fill_non_finite: [optional]
        The value to fill non-finite model flux values with.
        
    :param use_median_ratio: [optional]
        Use the median ratio of the observed flux to the model flux to estimate the best nodes.
        If `False`, then the mean ratio will be given.
        
    :param use_nmf_flux: [optional]
        Use the NMF-derived model flux to apply the filtering to (instead of the data).
    """

    from juliacall import Main as jl
    # TODO dev it instead
    jl.include(os.path.join(os.path.dirname(__file__), "Grok.jl"))
    
    if mask_path is not None:
        mask = np.loadtxt(expand_path(mask_path), dtype=bool)
    else:
        mask = None    
    
    if isinstance(spectra, ModelSelect):
        if limit is not None:
            spectra = spectra.limit(limit)
        # Note: if you don't use the `.iterator()` you may get out-of-memory issues from the GPU nodes 
        spectra = spectra.iterator()     
    
    fluxs, ivars, nmf_fluxs = ([], [], [])
    for spectrum in tqdm(spectra):
        
        flux, ivar = (spectrum.flux, spectrum.ivar)
                
        if mask is not None:
            ivar[~mask] = 0
        
        if inflate_errors:
            flux, ivar = inflate_errors_at_bad_pixels(flux, ivar, spectrum.pixel_flags, **kwargs)

        fluxs.append(flux)
        ivars.append(ivar)
        if use_nmf_flux:
            cf = np.copy(spectrum.nmf_rectified_model_flux * spectrum.continuum)
            cf[(~np.isfinite(cf)) | (cf <= 0)] = 1e-4            
            nmf_fluxs.append(cf)
        else:
            nmf_fluxs.append(flux)
        
    # supply a mask to Grok to ignore totally masked pixels
    if mask is None:
        mask = np.ones_like(flux, dtype=bool)

    if n_workers > 0:
        print("starting julia worker processes")
        jl.Grok.ensure_workers(n_workers)
            
    # Load the grid.
    print("loading grid")
    grid_labels, grid_points, grid_flux = grid = jl.Grok.load_grid(
        expand_path(grid_path), 
        use_subset=use_subset,
        fill_non_finite=fill_non_finite,
        v_sinis=v_sinis,
        ε=ε
    )
    print("loaded")

    results = jl.Grok.get_best_nodes(
        grid, 
        fluxs, 
        ivars, 
        nmf_fluxs, 
        use_median_ratio=use_median_ratio,
        refinement_levels=refinement_levels
    )

    for spectrum, result in zip(spectra, results):
        
        coarse_params, coarse_chi2, model_spectrum = result 
                
        result_dict = dict(
            source_pk=spectrum.source_pk,
            spectrum_pk=spectrum.spectrum_pk,
            coarse_chi2=coarse_chi2,
        )
        result_dict.update(dict(zip([f"coarse_{ln}" for ln in grid_labels], coarse_params)))
        yield Grok(**result_dict)


def inflate_errors_at_bad_pixels(
    flux,
    ivar,
    bitfield,
    skyline_ivar_multiplier=1e-4,
    bad_pixel_flux_value=1e-4,
    bad_pixel_ivar_value=1e-20,
    spike_threshold_to_inflate_uncertainty=3,
    max_ivar_value=400,
    **kwargs
):
    flux = np.copy(flux)
    ivar = np.copy(ivar)
    
    
    # Inflate errors around skylines,
    skyline_mask = (bitfield & 4096) > 0 # significant skyline
    ivar[skyline_mask] *= skyline_ivar_multiplier

    # Sometimes FERRE will run forever.
    if spike_threshold_to_inflate_uncertainty > 0:

        flux_median = np.nanmedian(flux)
        flux_stddev = np.nanstd(flux)

        delta = (flux - flux_median) / flux_stddev
        is_spike = (delta > spike_threshold_to_inflate_uncertainty)

        ivar[is_spike] = bad_pixel_ivar_value

    # Set bad pixels to have no useful data.
    if bad_pixel_flux_value is not None or bad_pixel_ivar_value is not None:                            
        bad = (
            ~np.isfinite(flux)
            | ~np.isfinite(ivar)
            | (flux < 0)
            | (ivar <= 0)
            | ((bitfield & 16639) > 0) # any bad value (level = 1)
        )
        flux[bad] = bad_pixel_flux_value
        ivar[bad] = bad_pixel_ivar_value        

    if max_ivar_value is not None:
        ivar = np.clip(ivar, 0, max_ivar_value)

    return (flux, ivar)



if __name__ == '__main__':
    # Let's do the kepler ajacent field
    from astra.models import ApogeeCoaddedSpectrumInApStar, Source
    sdss4_apogee_ids = """
2M18412508+4303240
2M18450005+4245104
2M18553575+4105089
2M18570444+4734401
2M19004810+3856578
2M19025959+4512417
2M19063516+3739380
2M19073581+4513065
2M19100248+3815049
2M19102221+4853112
2M19103347+4404049
2M19115040+4151422
2M19120422+4320515
2M19170003+3634201
2M19172983+4151321
2M19204102+3743025
2M19221025+4844196
2M19230991+3755447
2M19231009+4916084
2M19274170+4426119
2M19283274+3903403
2M19290483+4857332
2M19292004+3743086
2M19304193+4905158
2M19360994+4643236
2M19364294+4704099
2M19390540+3919325
2M19404406+4018111
2M19423341+3958080
2M19452903+4747473
2M19461201+4906083
2M19465588+5058197
2M19515772+4845561
2M19540916+4817228
2M19584964+4520550
2M20040830+4407511
2M18452786+4702029
2M18501911+4125493
2M18503405+4256469
2M18550270+4906586
2M18563593+4450339
2M18574686+4141355
2M18575358+4531335
2M18593221+4408012
2M19030398+4408067
2M19031936+4809465
2M19044822+4319420
2M19081716+3924583
2M19132893+4322431
2M19151548+4214292
2M19204818+4835597
2M19210586+4512568
2M19294869+3907574
2M19343888+4109436
2M19363934+4229515
2M19371573+3912149
2M19373289+4713273
2M19373952+4041325
2M19390407+4050174
2M19415508+3945414
2M19415577+4831280
2M19422693+4038597
2M19431187+4339204
2M19440075+4800542
2M19441934+4716319
2M19452399+4256525
2M19473449+5008477
2M19475031+4046557
2M19481124+4606317
2M19514644+4307245
2M19565074+4640081
2M18441040+4748003
2M18445792+4711423
2M18454273+4318127
2M18462545+4243415
2M18481466+4613467
2M18490564+4631406
2M18495269+4404330
2M18503900+4055336
2M18504192+4231150
2M18504794+4637102
2M18510530+4842191
2M18511240+4621571
2M18511762+4356107
2M18520168+4525456
2M18522187+4733526
2M18524057+4247316
2M18571567+4449312
2M18575188+4514236
2M18582947+4945410
2M18584335+4728453
2M18585634+3902433
2M18591247+4431443
2M19001868+4905365
2M19001953+3157462
2M19002258+3908388
2M19003472+3840264
2M19004734+4452598
2M19011630+4547575
2M19014130+3846502
2M19014504+4104500
2M19015301+4629499
2M19022731+4808344
2M19024826+4131208
2M19030263+4909150
2M19030505+3820112
2M19032243+4547495
2M19040628+4446240
2M19043294+4803072
2M19043306+4825465
2M19044782+4832359
2M19044946+4929242
2M19053308+4211201
2M19054621+4944573
2M19055045+4454531
2M19061346+3805157
2M19065745+3916557
2M19074435+4915418
2M19083567+3807130
2M19091490+4721514
2M19101609+4615464
2M19102806+4902583
2M19105779+3807097
2M19111922+5100573
2M19113802+4025211
2M19124367+4614462
2M19131544+3841358
2M19132148+4758028
2M19143032+4629187
2M19154301+4226056
2M19155843+3915396
2M19160443+4438315
2M19161229+4139153
2M19162872+3949410
2M19163533+4551582
2M19163913+3931378
2M19164680+3633087
2M19171645+3814212
2M19173233+3955170
2M19174572+4626262
2M19174597+5058481
2M19174796+3911136
2M19175303+3923229
2M19175396+4422570
2M19175683+3835482
2M19184961+3644552
2M19200957+3746128
2M19201451+4709502
2M19201752+3818216
2M19203536+4929354
2M19204958+4147429
2M19205061+4113122
2M19205079+3719502
2M19210086+3745339
2M19212498+4412311
2M19213924+4514037
2M19214069+3951121
2M19214952+4952463
2M19220642+3808347
2M19221650+4803104
2M19222812+5118100
2M19224278+3742265
2M19225313+4121554
2M19232348+3820389
2M19233322+3705216
2M19241926+4842071
2M19244484+4206243
2M19244544+3959308
2M19251994+4121340
2M19254347+4328340
2M19254724+4544081
2M19254744+4420124
2M19254848+4442241
2M19255838+3650557
2M19260282+4340555
2M19261281+3811549
2M19261331+3726425
2M19262450+3931415
2M19263228+3739240
2M19263604+3824337
2M19263652+5051045
2M19265205+3646419
2M19265403+4741293
2M19272447+4841572
2M19275747+3840145
2M19281827+3746296
2M19283383+3734463
2M19290017+4226343
2M19290899+3726467
2M19291108+3658492
2M19291298+3843589
2M19293125+4848253
2M19293224+4906198
2M19295717+4130562
2M19310386+3759524
2M19311651+4752470
2M19311845+3732331
2M19312163+3743519
2M19312521+4856205
2M19314738+4148344
2M19331236+4854457
2M19331823+4308005
2M19331912+4157188
2M19333557+4745414
2M19333652+4551394
2M19335277+4702446
2M19335606+4315249
2M19343935+4004385
2M19351595+4212449
2M19353743+4134535
2M19362997+4640224
2M19364511+3831345
2M19364769+3945288
2M19364843+4209232
2M19372616+3901541
2M19394010+4307580
2M19394925+3907298
2M19402202+4550341
2M19403511+4025267
2M19405136+5101517
2M19412774+5038199
2M19413087+4303501
2M19420328+4917050
2M19423985+4916476
2M19424465+4314494
2M19424968+5034286
2M19432965+4255038
2M19432982+4446533
2M19434843+3922148
2M19435695+4734226
2M19443415+5103342
2M19445366+3903441
2M19445512+4244466
2M19451260+4947453
2M19452145+4725114
2M19453774+4738433
2M19462259+4635079
2M19462864+5020257
2M19471461+4438144
2M19473339+4041137
2M19481305+4002492
2M19483057+4638589
2M19484843+4053327
2M19485620+4935581
2M19490851+4118201
2M19491051+4711206
2M19492618+4803404
2M19492762+4404308
2M19495818+4806343
2M19495994+4142273
2M19501406+4038048
2M19501586+3955332
2M19502722+4256144
2M19503629+4625100
2M19510302+4029223
2M19510892+4121279
2M19511688+4148231
2M19512634+4515545
2M19520159+4020107
2M19521906+4444467
2M19523384+4747072
2M19530511+4431004
2M19535404+4648150
2M19535559+4143549
2M19535708+4824384
2M19543671+4343229
2M19555743+4804378
2M19562019+4432525
2M19562576+4333125
2M19580025+4734486
2M19580481+4442212
2M19591565+4047014
2M20005213+4521219
2M20015171+4359017
2M20032827+4410523
2M20035411+4449059
2M20043893+4409009
    """.strip().split()
    
    # Let's do mean with NMF
    from astra.models.mwm import ApogeeCombinedSpectrum
    
    
    spectra = list(
        ApogeeCombinedSpectrum
        .select()
        .join(Source, on=(Source.pk == ApogeeCombinedSpectrum.source_pk))
        .where(Source.sdss4_apogee_id.in_(sdss4_apogee_ids))
    )
    
    results = list(grok(
        spectra,
        grid_path="/uufs/chpc.utah.edu/common/home/u6035527/gen_grid/big_grid_2024-12-20.h5",
        use_nmf_flux=True,
        use_subset=True,
        v_sinis=[0, 5, 7.5, 10, 20, 30, 50, 75, 100]
    ))
    
    import matplotlib.pyplot as plt
    
    # Match to ASPCAP results
    from astra.models import ASPCAP
    q = (
        ASPCAP
        .select()
        .where(ASPCAP.source_pk.in_([s.source_pk for s in spectra]))
    )
    aspcap_results = { ea.source_pk: ea for ea in q }
    
    def plot_comparison(ax_diff, ax, grok_results, aspcap_results, grok_label, aspcap_label, zlabel=None, **kwargs):
        
        y = np.array([(getattr(r, grok_label) or np.nan) for r in grok_results])
        x = np.array([(getattr(aspcap_results[r.source_pk], aspcap_label) or np.nan) for r in grok_results])

        if zlabel is None:
            z = None
            scatter_kwds = dict(facecolor="tab:blue", ec="k", lw=1)
            line_kwds = dict(c="#666666", lw=0.5, zorder=-1, ls=":")

        else:
            z = np.array([getattr(aspcap_results[r.source_pk], zlabel) for r in grok_results])
            scatter_kwds = dict(c=z, ec="k", lw=1)
            line_kwds = dict(c="#666666", lw=0.5, zorder=-1, ls=":")
        scatter_kwds.update(kwargs)
        
        scat = ax.scatter(x, y, **scatter_kwds)
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))
        ax.plot(limits, limits, **line_kwds)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        
        diff = np.array(y) - np.array(x)
        ax.set_xlabel(aspcap_label)
        ax.set_ylabel(grok_label)
        
        ax_diff.scatter(x, y - x, **scatter_kwds)
        ax_diff.axhline(0, **line_kwds)
        ax_diff.set_xticks([])
        ax_diff.set_xlim(limits)
        diff_limits = np.max(np.abs(ax_diff.get_ylim()))
        ax_diff.set_ylim(-diff_limits, +diff_limits)
        ax_diff.set_title(f"{np.nanmean(diff):.2f} +/- {np.nanstd(diff):.2f} ({np.sum(np.isfinite(diff))} spectra)")
        return scat
        
    L = 5
    fig, axes = plt.subplots(2, L, figsize=(4 * L, 4), gridspec_kw=dict(height_ratios=[1, 4]))
    zlabel = "snr"
    kwds = dict(vmin=0, vmax=200, s=25)
    plot_comparison(*axes.T[0], results, aspcap_results, "coarse_teff", "teff", zlabel, **kwds)
    plot_comparison(*axes.T[1], results, aspcap_results, "coarse_logg", "logg", zlabel, **kwds)
    scat = plot_comparison(*axes.T[2], results, aspcap_results, "coarse_m_h", "m_h_atm", zlabel, **kwds)
    plot_comparison(*axes.T[3], results, aspcap_results, "coarse_v_micro", "v_micro", zlabel, **kwds)
    plot_comparison(*axes.T[4], results, aspcap_results, "coarse_v_sini", "v_sini", zlabel, **kwds)
    #plot_comparison(*axes.T[5], results, aspcap_results, "coarse_c_m", "c_m_atm", zlabel, **kwds)
    #plot_comparison(*axes.T[6], results, aspcap_results, "coarse_n_m", "n_m_atm", zlabel, **kwds)
    
    if zlabel is not None:
        cbar = plt.colorbar(scat)
        cbar.set_label(zlabel)

    fig.tight_layout()
    output_path = expand_path("~/20240220_tmp3.png")
    fig.savefig(output_path)
    
    
    raise a
    
    # Now some other testing
    
    
    from astra.models.mwm import ApogeeCombinedSpectrum
    
    result, = grok([ApogeeCombinedSpectrum.get()])

    from astropy.table import Table
    t = Table.read(expand_path("~/apj514696t1_mrt_xm_aspcap.fits"))    
    is_measurement = (t["f_vsini"] != "<")
    t = t[is_measurement]
    
    sdss4_apogee_ids = ["2M" + ea[1:] for ea in t["2MASS"]]
    
    from astra.models import ApogeeCoaddedSpectrumInApStar, Source
    spectra = list(
        ApogeeCoaddedSpectrumInApStar
        .select()
        .join(Source)
        .where(
            Source.sdss4_apogee_id.in_(sdss4_apogee_ids)
        )        
    )
    
    results = list(grok(spectra))
    
    #paths = []
    #for twomass_id in t[is_measurement]["2MASS"]:
    #    apogee_id = twomass_id.lstrip("J")
    #    paths.append(f"tayar_2015/spectra/apStar-dr17-2M{apogee_id}.fits")
    
    
    from astra.products.pipeline_summary import create_astra_all_star_product, ignore_field_name_callable
    
    create_astra_all_star_product(
        Grok,
        apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    )

