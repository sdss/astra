"""
Additional test coverage for models that still have low or 0% coverage:
  - madgics.py (0%) - accessors, model classes, fields, path property
  - fields.py (0%) - deprecated wrapper module with custom field classes
  - best.py (0%) - has a missing Glossary import bug, handled gracefully
  - slam.py (78%) - flag combinations (flag_warn, flag_bad)
  - snow_white.py (84%) - remaining flag combinations, apply_noise_model reference
  - the_cannon.py (80%) - accessor classes, intermediate_output_path
  - apogee.py (82%) - path templates, flag properties, pad/field helpers
  - boss.py (87%) - path branches, pad_fieldid, isplate, field_group
"""
import os
os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

import numpy as np
import pytest
from astra.models.base import database


def _setup_tables(*models):
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    all_models = [Source, Spectrum] + list(models)
    database.create_tables(all_models)
    return Source, Spectrum


def _make_spectrum(Source, Spectrum):
    source_pk = Source.create().pk
    spectrum_pk = Spectrum.create().pk
    return source_pk, spectrum_pk


# ---------------------------------------------------------------------------
# madgics.py: accessors, model classes, fields, path property
# ---------------------------------------------------------------------------

class TestMadgicsAccessors:
    """Test the accessor classes defined in madgics.py."""

    def test_madgics_pixel_array_accessor_class_exists(self):
        from astra.models.madgics import MadgicsPixelArrayAccessor
        assert MadgicsPixelArrayAccessor is not None

    def test_resampled_madgics_pixel_array_accessor_class_exists(self):
        from astra.models.madgics import ResampledMadgicsPixelArrayAccessor
        assert ResampledMadgicsPixelArrayAccessor is not None

    def test_star_pickled_pixel_array_accessor_class_exists(self):
        from astra.models.madgics import StarPickledPixelArrayAccessor
        assert StarPickledPixelArrayAccessor is not None

    def test_madgics_accessor_inherits_base(self):
        from astra.models.madgics import MadgicsPixelArrayAccessor
        from astra.fields import BasePixelArrayAccessor
        assert issubclass(MadgicsPixelArrayAccessor, BasePixelArrayAccessor)

    def test_resampled_accessor_inherits_base(self):
        from astra.models.madgics import ResampledMadgicsPixelArrayAccessor
        from astra.fields import BasePixelArrayAccessor
        assert issubclass(ResampledMadgicsPixelArrayAccessor, BasePixelArrayAccessor)

    def test_star_pickled_accessor_inherits_base(self):
        from astra.models.madgics import StarPickledPixelArrayAccessor
        from astra.fields import BasePixelArrayAccessor
        assert issubclass(StarPickledPixelArrayAccessor, BasePixelArrayAccessor)


class TestMadgicsModels:
    """Test the MADGICS model classes."""

    def test_base_apogee_madgics_spectrum_has_fields(self):
        from astra.models.madgics import BaseApogeeMADGICSSpectrum
        model = BaseApogeeMADGICSSpectrum
        field_names = [f.name for f in model._meta.sorted_fields]
        for expected in ["release", "telescope", "field", "plate", "mjd", "fiber",
                         "map2visit", "map2star", "map2madgics", "rv_pixels", "rv_velocity"]:
            assert expected in field_names, f"Missing field: {expected}"

    def test_base_apogee_madgics_spectrum_path(self):
        from astra.models.madgics import BaseApogeeMADGICSSpectrum
        r = BaseApogeeMADGICSSpectrum()
        assert r.path == ""

    def test_base_apogee_madgics_meta_fields(self):
        from astra.models.madgics import BaseApogeeMADGICSSpectrum
        model = BaseApogeeMADGICSSpectrum
        field_names = [f.name for f in model._meta.sorted_fields]
        for expected in ["meta_apogee_id", "meta_ra", "meta_dec", "meta_glat", "meta_glon",
                         "meta_sfd", "meta_dr17_teff", "meta_dr17_logg", "meta_dr17_x_h",
                         "meta_dr17_vsini", "meta_drp_snr", "meta_drp_vhelio",
                         "meta_drp_vrel", "meta_drp_vrelerr", "meta_gaiaedr3_parallax",
                         "meta_gaiaedr3_source_id"]:
            assert expected in field_names, f"Missing meta field: {expected}"

    def test_theory_star_spectrum_path(self):
        from astra.models.madgics import ApMADGICSTheoryStarSpectrum
        r = ApMADGICSTheoryStarSpectrum()
        assert r.path == ""

    def test_data_driven_star_spectrum_path(self):
        from astra.models.madgics import ApMADGICSDataDrivenStarSpectrum
        r = ApMADGICSDataDrivenStarSpectrum()
        assert r.path == ""

    def test_theory_spectrum_filetype(self):
        from astra.models.madgics import ApMADGICSTheorySpectrum
        r = ApMADGICSTheorySpectrum()
        assert r.filetype == "apmadgics_th"

    def test_data_driven_spectrum_filetype(self):
        from astra.models.madgics import ApMADGICSDataDrivenSpectrum
        r = ApMADGICSDataDrivenSpectrum()
        assert r.filetype == "apmadgics_dd"

    def test_data_driven_visit_spectrum_filetype(self):
        from astra.models.madgics import ApMADGICSDataDrivenVisitSpectrum
        r = ApMADGICSDataDrivenVisitSpectrum()
        assert r.filetype == "apmadgics_dd_apVisit"

    def test_theory_star_has_flux_and_ivar(self):
        from astra.models.madgics import ApMADGICSTheoryStarSpectrum
        # These are PixelArrays defined as class attributes
        assert hasattr(ApMADGICSTheoryStarSpectrum, "flux")
        assert hasattr(ApMADGICSTheoryStarSpectrum, "ivar")

    def test_data_driven_star_has_flux_and_ivar(self):
        from astra.models.madgics import ApMADGICSDataDrivenStarSpectrum
        assert hasattr(ApMADGICSDataDrivenStarSpectrum, "flux")
        assert hasattr(ApMADGICSDataDrivenStarSpectrum, "ivar")

    def test_theory_spectrum_has_flux_and_ivar(self):
        from astra.models.madgics import ApMADGICSTheorySpectrum
        assert hasattr(ApMADGICSTheorySpectrum, "flux")
        assert hasattr(ApMADGICSTheorySpectrum, "ivar")

    def test_data_driven_spectrum_has_flux_and_ivar(self):
        from astra.models.madgics import ApMADGICSDataDrivenSpectrum
        assert hasattr(ApMADGICSDataDrivenSpectrum, "flux")
        assert hasattr(ApMADGICSDataDrivenSpectrum, "ivar")

    def test_theory_star_spectrum_fields(self):
        from astra.models.madgics import ApMADGICSTheoryStarSpectrum
        field_names = [f.name for f in ApMADGICSTheoryStarSpectrum._meta.sorted_fields]
        for expected in ["release", "telescope", "mean_fiber",
                         "meta_dr17_teff", "meta_dr17_logg", "meta_dr17_x_h"]:
            assert expected in field_names

    def test_data_driven_star_spectrum_fields(self):
        from astra.models.madgics import ApMADGICSDataDrivenStarSpectrum
        field_names = [f.name for f in ApMADGICSDataDrivenStarSpectrum._meta.sorted_fields]
        for expected in ["release", "telescope", "mean_fiber",
                         "meta_dr17_teff", "meta_dr17_logg", "meta_dr17_x_h"]:
            assert expected in field_names

    def test_madgics_accessor_get_returns_field_when_no_instance(self):
        from astra.models.madgics import MadgicsPixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = MadgicsPixelArrayAccessor(
            model=None, field=field, name="test",
            ext=None, column_name="test",
            path="/fake", key="fake_key"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_resampled_accessor_get_returns_field_when_no_instance(self):
        from astra.models.madgics import ResampledMadgicsPixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = ResampledMadgicsPixelArrayAccessor(
            model=None, field=field, name="test",
            ext=None, column_name="test",
            path="/fake", key="fake_key"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_star_pickled_accessor_get_returns_field_when_no_instance(self):
        from astra.models.madgics import StarPickledPixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = StarPickledPixelArrayAccessor(
            model=None, field=field, name="test",
            ext=None, column_name="test",
            path="/fake", key="fake_key"
        )
        result = accessor.__get__(None, None)
        assert result is field


# ---------------------------------------------------------------------------
# fields.py: deprecated wrapper module
# ---------------------------------------------------------------------------

class TestDeprecatedFields:
    """Test the deprecated fields.py wrapper module."""

    def test_import_prints_deprecation(self, capsys):
        """Importing astra.models.fields should print a deprecation message."""
        import importlib
        # Force re-import to trigger the print
        import astra.models.fields as mf
        importlib.reload(mf)
        captured = capsys.readouterr()
        assert "deprecated" in captured.out.lower()

    def test_autofield_exists(self):
        from astra.models.fields import AutoField
        assert AutoField is not None

    def test_integerfield_exists(self):
        from astra.models.fields import IntegerField
        assert IntegerField is not None

    def test_floatfield_exists(self):
        from astra.models.fields import FloatField
        assert FloatField is not None

    def test_bigintegerfield_exists(self):
        from astra.models.fields import BigIntegerField
        assert BigIntegerField is not None

    def test_smallintegerfield_exists(self):
        from astra.models.fields import SmallIntegerField
        assert SmallIntegerField is not None

    def test_datetimefield_exists(self):
        from astra.models.fields import DateTimeField
        assert DateTimeField is not None

    def test_booleanfield_exists(self):
        from astra.models.fields import BooleanField
        assert BooleanField is not None

    def test_bitfield_exists(self):
        from astra.models.fields import BitField
        assert BitField is not None

    def test_pixelarray_exists(self):
        from astra.models.fields import PixelArray
        assert PixelArray is not None

    def test_autofield_class_available(self):
        from astra.models.fields import AutoField
        # AutoField can't be instantiated unbound (no name attribute),
        # but verify the class exists and wraps peewee's AutoField
        from peewee import AutoField as _AutoField
        assert issubclass(AutoField, _AutoField)

    def test_integerfield_with_help_text(self):
        from astra.models.fields import IntegerField
        # Providing help_text skips the Glossary lookup (which needs self.name)
        f = IntegerField(help_text="test help")
        assert f.help_text == "test help"

    def test_floatfield_with_help_text(self):
        from astra.models.fields import FloatField
        f = FloatField(null=True, help_text="test")
        assert f is not None

    def test_bigintegerfield_with_help_text(self):
        from astra.models.fields import BigIntegerField
        f = BigIntegerField(help_text="test")
        assert f is not None

    def test_smallintegerfield_with_help_text(self):
        from astra.models.fields import SmallIntegerField
        f = SmallIntegerField(help_text="test")
        assert f is not None

    def test_datetimefield_with_help_text(self):
        from astra.models.fields import DateTimeField
        f = DateTimeField(help_text="test")
        assert f is not None

    def test_booleanfield_with_help_text(self):
        from astra.models.fields import BooleanField
        f = BooleanField(help_text="test")
        assert f is not None

    def test_integerfield_subclass(self):
        from astra.models.fields import IntegerField
        from peewee import IntegerField as _IntegerField
        assert issubclass(IntegerField, _IntegerField)

    def test_floatfield_subclass(self):
        from astra.models.fields import FloatField
        from peewee import FloatField as _FloatField
        assert issubclass(FloatField, _FloatField)

    def test_bigintegerfield_subclass(self):
        from astra.models.fields import BigIntegerField
        from peewee import BigIntegerField as _BigIntegerField
        assert issubclass(BigIntegerField, _BigIntegerField)

    def test_smallintegerfield_subclass(self):
        from astra.models.fields import SmallIntegerField
        from peewee import SmallIntegerField as _SmallIntegerField
        assert issubclass(SmallIntegerField, _SmallIntegerField)

    def test_datetimefield_subclass(self):
        from astra.models.fields import DateTimeField
        from peewee import DateTimeField as _DateTimeField
        assert issubclass(DateTimeField, _DateTimeField)

    def test_booleanfield_subclass(self):
        from astra.models.fields import BooleanField
        from peewee import BooleanField as _BooleanField
        assert issubclass(BooleanField, _BooleanField)

    def test_bitfield_default_zero(self):
        from astra.models.fields import BitField
        f = BitField()
        assert f.default == 0

    def test_bitfield_flag_auto_increment(self):
        from astra.models.fields import BitField
        f = BitField()
        f1 = f.flag()
        f2 = f.flag()
        # First flag value is 1, second is 2
        assert f1._value == 1
        assert f2._value == 2

    def test_bitfield_flag_explicit_value(self):
        from astra.models.fields import BitField
        f = BitField()
        flag = f.flag(value=4)
        assert flag._value == 4

    def test_bitfield_flag_help_text(self):
        from astra.models.fields import BitField
        f = BitField()
        flag = f.flag(help_text="A test flag")
        assert flag.help_text == "A test flag"

    def test_bitfield_flag_descriptor_set_invalid(self):
        from astra.models.fields import BitField
        f = BitField()
        flag = f.flag()
        # __set__ requires True or False
        class FakeInstance:
            pass
        inst = FakeInstance()
        inst.result_flags = 0
        # FlagDescriptor requires setting True or False
        with pytest.raises(ValueError, match="True or False"):
            flag.__set__(inst, "invalid")

    def test_base_pixel_array_accessor_set(self):
        from astra.models.fields import BasePixelArrayAccessor
        accessor = BasePixelArrayAccessor(
            model=None, field=None, name="test_flux",
            ext=None, column_name="test_flux"
        )

        class FakeInstance:
            pass

        inst = FakeInstance()
        accessor.__set__(inst, np.array([1.0, 2.0]))
        assert "test_flux" in inst.__pixel_data__
        np.testing.assert_array_equal(inst.__pixel_data__["test_flux"], [1.0, 2.0])

    def test_base_pixel_array_accessor_initialise(self):
        from astra.models.fields import BasePixelArrayAccessor
        accessor = BasePixelArrayAccessor(
            model=None, field=None, name="test",
            ext=None, column_name="test"
        )

        class FakeInstance:
            pass

        inst = FakeInstance()
        assert not hasattr(inst, "__pixel_data__")
        accessor._initialise_pixel_array(inst)
        assert hasattr(inst, "__pixel_data__")
        assert inst.__pixel_data__ == {}

    def test_base_pixel_array_accessor_initialise_idempotent(self):
        from astra.models.fields import BasePixelArrayAccessor
        accessor = BasePixelArrayAccessor(
            model=None, field=None, name="test",
            ext=None, column_name="test"
        )

        class FakeInstance:
            pass

        inst = FakeInstance()
        inst.__pixel_data__ = {"existing": 42}
        accessor._initialise_pixel_array(inst)
        # Should not overwrite existing data
        assert inst.__pixel_data__ == {"existing": 42}

    def test_log_lambda_array_accessor(self):
        from astra.models.fields import LogLambdaArrayAccessor
        accessor = LogLambdaArrayAccessor(
            model=None, field="dummy_field", name="wavelength",
            ext=None, column_name="wavelength",
            crval=4.179, cdelt=6e-6, naxis=10
        )
        # __get__ with no instance returns the field
        result = accessor.__get__(None, None)
        assert result == "dummy_field"

    def test_log_lambda_array_accessor_with_instance(self):
        from astra.models.fields import LogLambdaArrayAccessor
        accessor = LogLambdaArrayAccessor(
            model=None, field="dummy_field", name="wavelength",
            ext=None, column_name="wavelength",
            crval=4.179, cdelt=6e-6, naxis=10
        )

        class FakeInstance:
            pass

        inst = FakeInstance()
        result = accessor.__get__(inst, None)
        assert len(result) == 10
        expected = 10**(4.179 + 6e-6 * np.arange(10))
        np.testing.assert_allclose(result, expected)

    def test_pixel_array_field(self):
        from astra.models.fields import PixelArray, PixelArrayAccessorFITS
        pa = PixelArray(ext=1, column_name="flux")
        assert pa.ext == 1
        assert pa.column_name == "flux"
        assert pa.accessor_class is PixelArrayAccessorFITS

    def test_pixel_array_accessor_fits_no_instance(self):
        from astra.models.fields import PixelArrayAccessorFITS, PixelArray
        field = PixelArray()
        accessor = PixelArrayAccessorFITS(
            model=None, field=field, name="flux",
            ext=1, column_name="flux"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_pickled_pixel_array_accessor_no_instance(self):
        from astra.models.fields import PickledPixelArrayAccessor, PixelArray
        field = PixelArray()
        accessor = PickledPixelArrayAccessor(
            model=None, field=field, name="flux",
            ext=None, column_name="flux"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_pixel_array_accessor_hdf_no_instance(self):
        from astra.models.fields import PixelArrayAccessorHDF, PixelArray
        field = PixelArray()
        accessor = PixelArrayAccessorHDF(
            model=None, field=field, name="flux",
            ext=None, column_name="flux"
        )
        result = accessor.__get__(None, None)
        assert result is field


# ---------------------------------------------------------------------------
# best.py: import bug (missing Glossary import)
# ---------------------------------------------------------------------------

class TestBest:
    """Test best.py, which has a missing Glossary import."""

    def test_best_import_fails_with_name_error(self):
        """best.py uses Glossary without importing it, so it should raise NameError."""
        with pytest.raises(NameError):
            # Force a fresh import to trigger the NameError
            import importlib
            import sys
            # Remove cached module if present
            mod_name = "astra.models.best"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

    def test_best_module_has_glossary_bug(self):
        """Verify that best.py references Glossary without importing it."""
        import ast
        best_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "astra", "models", "best.py"
        )
        with open(best_path) as f:
            source = f.read()

        # Check the source uses Glossary
        assert "Glossary" in source

        # Check that Glossary is NOT imported
        tree = ast.parse(source)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name if alias.asname is None else alias.asname)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name if alias.asname is None else alias.asname)

        assert "Glossary" not in imported_names, "best.py now imports Glossary (bug may have been fixed)"


# ---------------------------------------------------------------------------
# slam.py: flag combinations (flag_warn, flag_bad)
# ---------------------------------------------------------------------------

class TestSlamFlags:
    """Test Slam flag_warn and flag_bad hybrid properties."""

    def setup_method(self):
        from astra.models.slam import Slam
        self.Slam = Slam
        _setup_tables(Slam)

    def _make_slam(self, **kwargs):
        Source, Spectrum = _setup_tables(self.Slam)
        source_pk, spectrum_pk = _make_spectrum(Source, Spectrum)

        class FakeSpectrum:
            pass

        s = FakeSpectrum()
        s.spectrum_pk = spectrum_pk
        s.source_pk = source_pk
        return self.Slam.from_spectrum(s, **kwargs)

    def test_flag_warn_no_flags(self):
        r = self.Slam()
        r.result_flags = 0
        assert not r.flag_warn

    def test_flag_warn_with_bad_optimizer(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_bad_optimizer_status = True
        assert r.flag_warn

    def test_flag_warn_with_outside_photometry(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_outside_photometry_range = True
        assert r.flag_warn

    def test_flag_bad_no_flags(self):
        r = self.Slam()
        r.result_flags = 0
        assert not r.flag_bad

    def test_flag_bad_with_teff_outside_bounds(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_teff_outside_bounds = True
        assert r.flag_bad

    def test_flag_bad_with_fe_h_outside_bounds(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_fe_h_outside_bounds = True
        assert r.flag_bad

    def test_flag_bad_with_outside_photometry(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_outside_photometry_range = True
        assert r.flag_bad

    def test_flag_bad_with_bad_optimizer(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_bad_optimizer_status = True
        assert r.flag_bad

    def test_flag_bad_with_all_bad_flags(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_teff_outside_bounds = True
        r.flag_fe_h_outside_bounds = True
        r.flag_outside_photometry_range = True
        r.flag_bad_optimizer_status = True
        assert r.flag_bad
        assert r.flag_warn

    def test_slam_fields_exist(self):
        field_names = [f.name for f in self.Slam._meta.sorted_fields]
        for expected in ["teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h",
                         "fe_h_niu", "e_fe_h_niu", "alpha_fe", "e_alpha_fe",
                         "chi2", "rchi2", "success", "status", "optimality"]:
            assert expected in field_names

    def test_slam_correlation_fields(self):
        field_names = [f.name for f in self.Slam._meta.sorted_fields]
        for expected in ["rho_teff_logg", "rho_teff_fe_h", "rho_teff_fe_h_niu",
                         "rho_teff_alpha_fe", "rho_logg_fe_h_niu",
                         "rho_logg_alpha_fe", "rho_logg_fe_h",
                         "rho_fe_h_fe_h_niu", "rho_fe_h_alpha_fe"]:
            assert expected in field_names

    def test_slam_initial_fields(self):
        field_names = [f.name for f in self.Slam._meta.sorted_fields]
        for expected in ["initial_teff", "initial_logg", "initial_fe_h",
                         "initial_alpha_fe", "initial_fe_h_niu"]:
            assert expected in field_names

    def test_slam_pixel_array_class(self):
        from astra.models.slam import SlamPixelArray, SlamPixelArrayAccessor
        pa = SlamPixelArray()
        assert pa.accessor_class is SlamPixelArrayAccessor

    def test_slam_pixel_accessor_no_instance(self):
        from astra.models.slam import SlamPixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = SlamPixelArrayAccessor(
            model=None, field=field, name="model_flux",
            ext=None, column_name="model_flux"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_slam_intermediate_output_path(self):
        r = self.Slam()
        r.spectrum_pk = 12345
        r.v_astra = 100
        path = r.intermediate_output_path
        assert "slam" in path
        assert "12345" in path

    def test_slam_flag_not_magnitude_cut(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_not_magnitude_cut = True
        assert r.flag_not_magnitude_cut
        # not_magnitude_cut is not in flag_bad or flag_warn
        assert not r.flag_bad
        assert not r.flag_warn

    def test_slam_flag_not_carton_match(self):
        r = self.Slam()
        r.result_flags = 0
        r.flag_not_carton_match = True
        assert r.flag_not_carton_match
        assert not r.flag_bad
        assert not r.flag_warn


# ---------------------------------------------------------------------------
# snow_white.py: remaining flag combinations
# ---------------------------------------------------------------------------

class TestSnowWhiteFlags:
    """Test SnowWhite flag combinations and properties."""

    def setup_method(self):
        from astra.models.snow_white import SnowWhite
        self.SnowWhite = SnowWhite
        _setup_tables(SnowWhite)

    def test_no_flags_set(self):
        r = self.SnowWhite()
        r.result_flags = 0
        assert not r.flag_low_snr
        assert not r.flag_unconverged
        assert not r.flag_teff_grid_edge_bad
        assert not r.flag_logg_grid_edge_bad
        assert not r.flag_no_flux
        assert not r.flag_not_mwm_wd
        assert not r.flag_missing_bp_rp_mag

    def test_flag_low_snr(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_low_snr = True
        assert r.flag_low_snr
        assert not r.flag_unconverged

    def test_flag_unconverged(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_unconverged = True
        assert r.flag_unconverged

    def test_flag_teff_grid_edge_bad(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_teff_grid_edge_bad = True
        assert r.flag_teff_grid_edge_bad

    def test_flag_logg_grid_edge_bad(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_logg_grid_edge_bad = True
        assert r.flag_logg_grid_edge_bad

    def test_flag_no_flux(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_no_flux = True
        assert r.flag_no_flux

    def test_flag_not_mwm_wd(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_not_mwm_wd = True
        assert r.flag_not_mwm_wd

    def test_flag_missing_bp_rp_mag(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_missing_bp_rp_mag = True
        assert r.flag_missing_bp_rp_mag

    def test_multiple_flags(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_low_snr = True
        r.flag_unconverged = True
        r.flag_no_flux = True
        assert r.flag_low_snr
        assert r.flag_unconverged
        assert r.flag_no_flux
        assert not r.flag_teff_grid_edge_bad

    def test_clear_flag(self):
        r = self.SnowWhite()
        r.result_flags = 0
        r.flag_low_snr = True
        assert r.flag_low_snr
        r.flag_low_snr = False
        assert not r.flag_low_snr

    def test_classification_fields(self):
        field_names = [f.name for f in self.SnowWhite._meta.sorted_fields]
        for expected in ["classification", "p_cv", "p_da", "p_dab", "p_dabz",
                         "p_dah", "p_dahe", "p_dao", "p_daz", "p_da_ms",
                         "p_db", "p_dba", "p_dbaz", "p_dbh", "p_dbz",
                         "p_db_ms", "p_dc", "p_dc_ms", "p_do", "p_dq",
                         "p_dqz", "p_dqpec", "p_dz", "p_dza", "p_dzb",
                         "p_dzba", "p_mwd", "p_hotdq"]:
            assert expected in field_names

    def test_stellar_params_fields(self):
        field_names = [f.name for f in self.SnowWhite._meta.sorted_fields]
        for expected in ["teff", "e_teff", "logg", "e_logg", "v_rel",
                         "raw_e_teff", "raw_e_logg"]:
            assert expected in field_names

    def test_intermediate_pixel_array_class(self):
        from astra.models.snow_white import IntermediatePixelArray, IntermediatePixelArrayAccessor
        pa = IntermediatePixelArray(ext=1)
        assert pa.accessor_class is IntermediatePixelArrayAccessor

    def test_intermediate_accessor_no_instance(self):
        from astra.models.snow_white import IntermediatePixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = IntermediatePixelArrayAccessor(
            model=None, field=field, name="model_flux",
            ext=1, column_name="model_flux"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_apply_noise_model_function_exists(self):
        from astra.models.snow_white import apply_noise_model
        assert callable(apply_noise_model)


# ---------------------------------------------------------------------------
# the_cannon.py: accessor classes, paths, fields
# ---------------------------------------------------------------------------

class TestTheCannonExtended:
    """Test TheCannon accessor classes and additional paths."""

    def test_cannon_pixel_array_accessor_class(self):
        from astra.models.the_cannon import TheCannonPixelArrayAccessor
        from astra.fields import BasePixelArrayAccessor
        assert issubclass(TheCannonPixelArrayAccessor, BasePixelArrayAccessor)

    def test_cannon_pixel_array_class(self):
        from astra.models.the_cannon import TheCannonPixelArray, TheCannonPixelArrayAccessor
        pa = TheCannonPixelArray()
        assert pa.accessor_class is TheCannonPixelArrayAccessor

    def test_cannon_accessor_no_instance(self):
        from astra.models.the_cannon import TheCannonPixelArrayAccessor
        from astra.fields import PixelArray
        field = PixelArray()
        accessor = TheCannonPixelArrayAccessor(
            model=None, field=field, name="model_flux",
            ext=None, column_name="model_flux"
        )
        result = accessor.__get__(None, None)
        assert result is field

    def test_cannon_fields(self):
        from astra.models.the_cannon import TheCannon
        field_names = [f.name for f in TheCannon._meta.sorted_fields]
        for expected in ["teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h",
                         "v_micro", "e_v_micro", "v_macro", "e_v_macro",
                         "chi2", "rchi2", "ier", "nfev", "x0_index"]:
            assert expected in field_names

    def test_cannon_abundance_fields(self):
        from astra.models.the_cannon import TheCannon
        field_names = [f.name for f in TheCannon._meta.sorted_fields]
        for element in ["c_fe", "n_fe", "o_fe", "na_fe", "mg_fe", "al_fe",
                        "si_fe", "s_fe", "k_fe", "ca_fe", "ti_fe", "v_fe",
                        "cr_fe", "mn_fe", "ni_fe"]:
            assert element in field_names
            assert f"e_{element}" in field_names
            assert f"raw_e_{element}" in field_names

    def test_cannon_raw_error_fields(self):
        from astra.models.the_cannon import TheCannon
        field_names = [f.name for f in TheCannon._meta.sorted_fields]
        for expected in ["raw_e_teff", "raw_e_logg", "raw_e_fe_h",
                         "raw_e_v_micro", "raw_e_v_macro"]:
            assert expected in field_names

    def test_cannon_flag_fitting_failure(self):
        from astra.models.the_cannon import TheCannon
        r = TheCannon()
        r.result_flags = 0
        assert not r.flag_fitting_failure
        r.flag_fitting_failure = True
        assert r.flag_fitting_failure

    def test_cannon_intermediate_output_path_short_pk(self):
        from astra.models.the_cannon import TheCannon
        r = TheCannon()
        r.source_pk = 5
        r.spectrum_pk = 3
        r.v_astra = 1
        path = r.intermediate_output_path
        # "5"[-4:-2] = "", "5"[-2:] = "5"
        assert path == "$MWM_ASTRA/1/pipelines/TheCannon//5/5-3.pkl"

    def test_set_formal_errors_function_exists(self):
        from astra.models.the_cannon import set_formal_errors
        assert callable(set_formal_errors)

    def test_apply_noise_model_function_exists(self):
        from astra.models.the_cannon import apply_noise_model
        assert callable(apply_noise_model)


# ---------------------------------------------------------------------------
# apogee.py: path templates, flag properties, transform functions
# ---------------------------------------------------------------------------

class TestApogeeVisitSpectrumPaths:
    """Test ApogeeVisitSpectrum path generation."""

    def test_sdss5_path_template(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        template = ApogeeVisitSpectrum.get_path_template("sdss5", "apo25m")
        assert "ipl-5" in template
        assert "{apred}" in template
        assert "{telescope}" in template

    def test_dr17_apo1m_path_template(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        template = ApogeeVisitSpectrum.get_path_template("dr17", "apo1m")
        assert "dr17" in template
        assert "{reduction}" in template
        assert "apo1m" not in template  # telescope is a placeholder

    def test_dr17_apo25m_path_template(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        template = ApogeeVisitSpectrum.get_path_template("dr17", "apo25m")
        assert "dr17" in template
        assert "{plate}" in template
        assert "{fiber:0>3}" in template

    def test_visit_spectrum_path_property(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.release = "sdss5"
        r.apred = "daily"
        r.telescope = "apo25m"
        r.field = "100"
        r.plate = "1000"
        r.mjd = 59000
        r.fiber = 42
        r.prefix = "ap"
        r.reduction = ""
        path = r.path
        assert "ipl-5" in path
        assert "daily" in path
        assert "59000" in path
        assert "042" in path  # fiber is zero-padded to 3

    def test_visit_spectrum_flag_bad(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        assert not r.flag_bad

    def test_visit_spectrum_flag_bad_with_bad_pixels(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_bad_pixels = True
        assert r.flag_bad

    def test_visit_spectrum_flag_bad_with_very_bright_neighbor(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_very_bright_neighbor = True
        assert r.flag_bad

    def test_visit_spectrum_flag_bad_with_bad_rv(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_bad_rv_combination = True
        assert r.flag_bad

    def test_visit_spectrum_flag_bad_with_rv_failure(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_rv_failure = True
        assert r.flag_bad

    def test_visit_spectrum_flag_warn_no_flags(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        assert not r.flag_warn

    def test_visit_spectrum_flag_warn_any_flag(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_commissioning = True
        assert r.flag_warn

    def test_visit_spectrum_many_flags(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        # Set multiple flags
        r.flag_bad_pixels = True
        r.flag_commissioning = True
        r.flag_bright_neighbor = True
        r.flag_low_snr = True
        assert r.flag_bad_pixels
        assert r.flag_commissioning
        assert r.flag_bright_neighbor
        assert r.flag_low_snr
        assert r.flag_bad  # bad_pixels triggers flag_bad
        assert r.flag_warn  # any flag triggers flag_warn

    def test_visit_spectrum_persist_flags(self):
        from astra.models.apogee import ApogeeVisitSpectrum
        r = ApogeeVisitSpectrum()
        r.visit_flags = 0
        r.flag_persist_high = True
        assert r.flag_persist_high
        assert r.flag_warn
        assert not r.flag_bad  # persist flags don't trigger flag_bad

    def test_transform_err_to_ivar(self):
        from astra.models.apogee import _transform_err_to_ivar
        err = np.array([[2.0, 0.5, 0.0]])
        ivar = _transform_err_to_ivar(err)
        assert ivar[0] == pytest.approx(0.25)
        assert ivar[1] == pytest.approx(4.0)
        assert ivar[2] == 0.0  # 0 error -> inf -> clipped to 0


class TestApogeeVisitInApStarPaths:
    """Test ApogeeVisitSpectrumInApStar path generation."""

    def test_dr17_path(self):
        from astra.models.apogee import ApogeeVisitSpectrumInApStar
        r = ApogeeVisitSpectrumInApStar()
        r.release = "dr17"
        r.apred = "r13"
        r.apstar = "stars"
        r.obj = "2M00000000+0000000"
        r.telescope = "apo25m"
        r.field = "000+00"
        r.prefix = "ap"
        r.plate = "1000"
        r.mjd = 59000
        r.fiber = 42
        r.healpix = None
        path = r.path
        assert "dr17" in path
        assert "r13" in path
        assert "apStar" in path or "Star" in path


class TestApogeeCoadded:
    """Test ApogeeCoaddedSpectrumInApStar."""

    def test_coadded_dr17_path(self):
        from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
        r = ApogeeCoaddedSpectrumInApStar()
        r.release = "dr17"
        r.apred = "r13"
        r.apstar = "stars"
        r.obj = "2M00000000+0000000"
        r.telescope = "apo25m"
        r.field = "000+00"
        r.prefix = "ap"
        r.healpix = None
        path = r.path
        assert "dr17" in path
        assert "r13" in path

    def test_coadded_sdss5_path(self):
        from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
        r = ApogeeCoaddedSpectrumInApStar()
        r.release = "sdss5"
        r.apred = "daily"
        r.apstar = "stars"
        r.obj = "2M00000000+0000000"
        r.telescope = "apo25m"
        r.healpix = 12345
        r.field = ""
        r.prefix = ""
        path = r.path
        assert "ipl-5" in path
        assert "12345" in path
        assert "12" in path  # healpix_group = 12345 // 1000 = 12

    def test_coadded_fields(self):
        from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
        field_names = [f.name for f in ApogeeCoaddedSpectrumInApStar._meta.sorted_fields]
        for expected in ["release", "apred", "apstar", "obj", "telescope",
                         "healpix", "field", "prefix", "min_mjd", "max_mjd",
                         "n_entries", "n_visits", "n_good_visits", "n_good_rvs",
                         "snr", "mean_fiber", "std_fiber", "v_rad", "e_v_rad",
                         "std_v_rad", "median_e_v_rad"]:
            assert expected in field_names

    def test_transform_coadded_spectrum(self):
        from astra.models.apogee import _transform_coadded_spectrum

        class FakeImage:
            pass

        class FakeInstance:
            pass

        v_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _transform_coadded_spectrum(v_2d, FakeImage(), FakeInstance())
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_transform_coadded_spectrum_1d(self):
        from astra.models.apogee import _transform_coadded_spectrum

        v_1d = np.array([1.0, 2.0, 3.0])
        result = _transform_coadded_spectrum(v_1d, None, None)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# boss.py: path branches, pad_fieldid, isplate, field_group
# ---------------------------------------------------------------------------

class TestBossVisitSpectrumPaths:
    """Test BossVisitSpectrum path generation and helpers."""

    def test_path_v6_2_1_with_spec_file(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_2_1"
        r.fieldid = 15000
        r.mjd = 59000
        r.catalogid = 12345
        r.spec_file = "spec-015000-59000-12345.fits"
        path = r.path
        assert "ipl-5" in path
        assert "spec-015000-59000-12345.fits" in path

    def test_path_v6_2_1_without_spec_file(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_2_1"
        r.fieldid = 15000
        r.mjd = 59000
        r.catalogid = 12345
        r.spec_file = None
        path = r.path
        assert "ipl-5" in path
        assert "spec-015000-59000-12345.fits" in path

    def test_path_v6_2_x(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_2_0"
        r.fieldid = 15000
        r.mjd = 59000
        r.catalogid = 12345
        r.spec_file = None
        path = r.path
        assert "sdsswork" in path
        assert "daily" in path

    def test_path_v6_0_1(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_1"
        r.fieldid = 15000
        r.mjd = 59000
        r.catalogid = 12345
        r.spec_file = None
        path = r.path
        assert "sdsswork" in path
        assert "full" in path
        # v6_0_1 does NOT use daily path
        assert "daily" not in path

    def test_pad_fieldid_v6_2_1(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_2_1"
        r.fieldid = 123
        assert r.pad_fieldid == "000123"

    def test_pad_fieldid_v6_0_1(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_1"
        r.fieldid = 123
        assert r.pad_fieldid == "123"

    def test_pad_fieldid_v6_0_2(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_2"
        r.fieldid = 456
        assert r.pad_fieldid == "456"

    def test_pad_fieldid_v6_0_3(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_3"
        r.fieldid = 789
        assert r.pad_fieldid == "789"

    def test_pad_fieldid_v6_0_4(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_4"
        r.fieldid = 10
        assert r.pad_fieldid == "10"

    def test_isplate_v6_0_1(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_1"
        assert r.isplate == "p"

    def test_isplate_v6_0_2(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_2"
        assert r.isplate == "p"

    def test_isplate_v6_0_3(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_3"
        assert r.isplate == "p"

    def test_isplate_v6_0_4(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_0_4"
        assert r.isplate == "p"

    def test_isplate_v6_2_1(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = "v6_2_1"
        assert r.isplate == ""

    def test_isplate_empty_run2d(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.run2d = ""
        assert r.isplate == ""

    def test_field_group(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.fieldid = 15000
        assert r.field_group == "015XXX"

    def test_field_group_small(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.fieldid = 500
        assert r.field_group == "000XXX"

    def test_field_group_large(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.fieldid = 123456
        assert r.field_group == "123XXX"

    def test_e_flux_property(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.ivar = np.array([4.0, 1.0, 0.0])
        e_flux = r.e_flux
        assert e_flux[0] == pytest.approx(0.5)
        assert e_flux[1] == pytest.approx(1.0)
        assert np.isinf(e_flux[2])

    def test_boss_fields_exist(self):
        from astra.models.boss import BossVisitSpectrum
        field_names = [f.name for f in BossVisitSpectrum._meta.sorted_fields]
        for expected in ["release", "run2d", "mjd", "fieldid", "catalogid",
                         "healpix", "spec_file", "n_exp", "exptime",
                         "snr", "telescope"]:
            assert expected in field_names

    def test_boss_observing_fields(self):
        from astra.models.boss import BossVisitSpectrum
        field_names = [f.name for f in BossVisitSpectrum._meta.sorted_fields]
        for expected in ["alt", "az", "seeing", "airmass", "airtemp",
                         "dewpoint", "humidity", "pressure"]:
            assert expected in field_names

    def test_boss_gri_gaia_transform_flags(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.gri_gaia_transform_flags = 0
        assert not r.flag_u_gaia_transformed
        r.flag_u_gaia_transformed = True
        assert r.flag_u_gaia_transformed

    def test_boss_zwarning_flags(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.zwarning_flags = 0
        assert not r.flag_sky_fiber
        r.flag_sky_fiber = True
        assert r.flag_sky_fiber

    def test_boss_zwarning_multiple_flags(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.zwarning_flags = 0
        r.flag_sky_fiber = True
        r.flag_little_wavelength_coverage = True
        r.flag_unplugged = True
        assert r.flag_sky_fiber
        assert r.flag_little_wavelength_coverage
        assert r.flag_unplugged
        assert not r.flag_small_delta_chi2

    def test_boss_gri_gaia_multiple_flags(self):
        from astra.models.boss import BossVisitSpectrum
        r = BossVisitSpectrum()
        r.gri_gaia_transform_flags = 0
        r.flag_g_gaia_transformed = True
        r.flag_r_gaia_transformed = True
        r.flag_position_offset = True
        assert r.flag_g_gaia_transformed
        assert r.flag_r_gaia_transformed
        assert r.flag_position_offset
        assert not r.flag_u_gaia_transformed

    def test_boss_xcsao_fields(self):
        from astra.models.boss import BossVisitSpectrum
        field_names = [f.name for f in BossVisitSpectrum._meta.sorted_fields]
        for expected in ["xcsao_v_rad", "xcsao_e_v_rad", "xcsao_teff",
                         "xcsao_e_teff", "xcsao_logg", "xcsao_e_logg",
                         "xcsao_fe_h", "xcsao_e_fe_h", "xcsao_rxc"]:
            assert expected in field_names
