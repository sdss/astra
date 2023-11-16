from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    DateTimeField
)
import datetime
from playhouse.postgres_ext import ArrayField

from astra import __version__
from astra.glossary import Glossary
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin


class LineForest(BaseModel, PipelineOutputMixin):

    """A result from the LineForest pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_pk
    )
    
    #> Astra Metadata
    task_pk = AutoField(help_text=Glossary.task_pk)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True, help_text=Glossary.t_overhead)
    tag = TextField(default="", index=True, help_text=Glossary.tag)
    
    #> H-alpha (6562.8 +/- 200A)
    eqw_h_alpha = FloatField(null=True, help_text="Equivalent width of H-alpha [A]")
    abs_h_alpha = FloatField(null=True)
    detection_stat_h_alpha = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_alpha = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_alpha = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_alpha = ArrayField(FloatField, null=True)

    #> H-beta (4861.3 +/- 200 A)
    eqw_h_beta = FloatField(null=True, help_text="Equivalent width of H-beta [A]")
    abs_h_beta = FloatField(null=True)
    detection_stat_h_beta = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_beta = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_beta = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_beta = ArrayField(FloatField, null=True)

    #> H-gamma (4340.5 +/- 200 A)
    eqw_h_gamma = FloatField(null=True, help_text="Equivalent width of H-gamma [A]")
    abs_h_gamma = FloatField(null=True)
    detection_stat_h_gamma = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_gamma = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_gamma = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_gamma = ArrayField(FloatField, null=True)

    #> H-delta (4101.7 +/- 200 A)
    eqw_h_delta = FloatField(null=True, help_text="Equivalent width of H-delta [A]")
    abs_h_delta = FloatField(null=True)
    detection_stat_h_delta = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_delta = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_delta = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_delta = ArrayField(FloatField, null=True)

    #> H-epsilon (3970.1 +/- 200 A)
    eqw_h_epsilon = FloatField(null=True, help_text="Equivalent width of H-epsilon [A]")
    abs_h_epsilon = FloatField(null=True)
    detection_stat_h_epsilon = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_epsilon = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_epsilon = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_epsilon = ArrayField(FloatField, null=True)

    #> H-8 (3889.064 +/- 200 A)
    eqw_h_8 = FloatField(null=True, help_text="Equivalent width of H-8 [A]")
    abs_h_8 = FloatField(null=True)
    detection_stat_h_8 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_8 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_8 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_8 = ArrayField(FloatField, null=True)

    #> H-9 (3835.391 +/- 200 A)
    eqw_h_9 = FloatField(null=True, help_text="Equivalent width of H-9 [A]")
    abs_h_9 = FloatField(null=True)
    detection_stat_h_9 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_9 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_9 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_9 = ArrayField(FloatField, null=True)

    #> H-10 (3797.904 +/- 200 A)
    eqw_h_10 = FloatField(null=True, help_text="Equivalent width of H-10 [A]")
    abs_h_10 = FloatField(null=True)
    detection_stat_h_10 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_10 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_10 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_10 = ArrayField(FloatField, null=True)
    
    #> H-11 (3770.637 +/- 200 A)
    eqw_h_11 = FloatField(null=True, help_text="Equivalent width of H-11 [A]")
    abs_h_11 = FloatField(null=True)
    detection_stat_h_11 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_11 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_11 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_11 = ArrayField(FloatField, null=True)

    #> H-12 (3750.158 +/- 50 A)
    eqw_h_12 = FloatField(null=True, help_text="Equivalent width of H-12 [A]")
    abs_h_12 = FloatField(null=True)
    detection_stat_h_12 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_12 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_12 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_12 = ArrayField(FloatField, null=True)

    #> H-13 (3734.369 +/- 50 A)
    eqw_h_13 = FloatField(null=True, help_text="Equivalent width of H-13 [A]")
    abs_h_13 = FloatField(null=True)
    detection_stat_h_13 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_13 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_13 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_13 = ArrayField(FloatField, null=True)

    #> H-14 (3721.945 +/- 50 A)
    eqw_h_14 = FloatField(null=True, help_text="Equivalent width of H-14 [A]")
    abs_h_14 = FloatField(null=True)
    detection_stat_h_14 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_14 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_14 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_14 = ArrayField(FloatField, null=True)

    #> H-15 (3711.977 +/- 50 A)
    eqw_h_15 = FloatField(null=True, help_text="Equivalent width of H-15 [A]")
    abs_h_15 = FloatField(null=True)
    detection_stat_h_15 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_15 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_15 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_15 = ArrayField(FloatField, null=True)
    
    #> H-16 (3703.859 +/- 50 A)
    eqw_h_16 = FloatField(null=True, help_text="Equivalent width of H-16 [A]")
    abs_h_16 = FloatField(null=True)
    detection_stat_h_16 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_16 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_16 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_16 = ArrayField(FloatField, null=True)

    #> H-17 (3697.157 +/- 50 A)
    eqw_h_17 = FloatField(null=True, help_text="Equivalent width of H-17 [A]")
    abs_h_17 = FloatField(null=True)
    detection_stat_h_17 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_h_17 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_h_17 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_h_17 = ArrayField(FloatField, null=True)

    #> Pa-7 (10049.4889 +/- 200 A)
    eqw_pa_7 = FloatField(null=True, help_text="Equivalent width of Pa-7 [A]")
    abs_pa_7 = FloatField(null=True)
    detection_stat_pa_7 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_7 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_7 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_7 = ArrayField(FloatField, null=True)

    #> Pa-8 (9546.0808 +/- 200 A)
    eqw_pa_8 = FloatField(null=True, help_text="Equivalent width of Pa-8 [A]")
    abs_pa_8 = FloatField(null=True)
    detection_stat_pa_8 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_8 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_8 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_8 = ArrayField(FloatField, null=True)

    #> Pa-9 (9229.12 +/- 200 A)
    eqw_pa_9 = FloatField(null=True, help_text="Equivalent width of Pa-9 [A]")
    abs_pa_9 = FloatField(null=True)
    detection_stat_pa_9 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_9 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_9 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_9 = ArrayField(FloatField, null=True)

    #> Pa-10 (9014.909 +/- 200 A)
    eqw_pa_10 = FloatField(null=True, help_text="Equivalent width of Pa-10 [A]")
    abs_pa_10 = FloatField(null=True)
    detection_stat_pa_10 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_10 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_10 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_10 = ArrayField(FloatField, null=True)

    #> Pa-11 (8862.782 +/- 200 A)
    eqw_pa_11 = FloatField(null=True, help_text="Equivalent width of Pa-11 [A]")
    abs_pa_11 = FloatField(null=True)
    detection_stat_pa_11 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_11 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_11 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_11 = ArrayField(FloatField, null=True)

    #> Pa-12 (8750.472 +/- 200 A)
    eqw_pa_12 = FloatField(null=True, help_text="Equivalent width of Pa-12 [A]")
    abs_pa_12 = FloatField(null=True)
    detection_stat_pa_12 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_12 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_12 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_12 = ArrayField(FloatField, null=True)

    #> Pa-13 (8665.019 +/- 200 A)
    eqw_pa_13 = FloatField(null=True, help_text="Equivalent width of Pa-13 [A]")
    abs_pa_13 = FloatField(null=True)
    detection_stat_pa_13 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_13 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_13 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_13 = ArrayField(FloatField, null=True)

    #> Pa-14 (8598.392 +/- 200 A)
    eqw_pa_14 = FloatField(null=True, help_text="Equivalent width of Pa-14 [A]")
    abs_pa_14 = FloatField(null=True)
    detection_stat_pa_14 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_14 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_14 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_14 = ArrayField(FloatField, null=True)

    #> Pa-15 (8545.383 +/- 200 A)
    eqw_pa_15 = FloatField(null=True, help_text="Equivalent width of Pa-15 [A]")
    abs_pa_15 = FloatField(null=True)
    detection_stat_pa_15 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_15 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_15 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_15 = ArrayField(FloatField, null=True)

    #> Pa-16 (8502.483 +/- 200 A)
    eqw_pa_16 = FloatField(null=True, help_text="Equivalent width of Pa-16 [A]")
    abs_pa_16 = FloatField(null=True)
    detection_stat_pa_16 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_16 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_16 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_16 = ArrayField(FloatField, null=True)

    #> Pa-17 (8467.254 +/- 200 A)
    eqw_pa_17 = FloatField(null=True, help_text="Equivalent width of Pa-17 [A]")
    abs_pa_17 = FloatField(null=True)
    detection_stat_pa_17 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_pa_17 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_pa_17 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_pa_17 = ArrayField(FloatField, null=True)

    #> Ca II (8662.14 +/- 50 A)
    eqw_ca_ii_8662 = FloatField(null=True, help_text="Equivalent width of Ca II at 8662 A [A]")
    abs_ca_ii_8662 = FloatField(null=True)
    detection_stat_ca_ii_8662 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_ca_ii_8662 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_ca_ii_8662 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_ca_ii_8662 = ArrayField(FloatField, null=True)

    #> Ca II (8542.089 +/- 50 A)
    eqw_ca_ii_8542 = FloatField(null=True, help_text="Equivalent width of Ca II at 8542 A [A]")
    abs_ca_ii_8542 = FloatField(null=True)
    detection_stat_ca_ii_8542 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_ca_ii_8542 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_ca_ii_8542 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_ca_ii_8542 = ArrayField(FloatField, null=True)

    #> Ca II (8498.018 +/- 50 A)
    eqw_ca_ii_8498 = FloatField(null=True, help_text="Equivalent width of Ca II at 8498 A [A]")
    abs_ca_ii_8498 = FloatField(null=True)
    detection_stat_ca_ii_8498 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_ca_ii_8498 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_ca_ii_8498 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_ca_ii_8498 = ArrayField(FloatField, null=True)

    #> Ca K (3933.6614 +/- 200 A)
    eqw_ca_k_3933 = FloatField(null=True, help_text="Equivalent width of Ca K at 3933 A [A]")
    abs_ca_k_3933 = FloatField(null=True)
    detection_stat_ca_k_3933 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_ca_k_3933 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_ca_k_3933 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_ca_k_3933 = ArrayField(FloatField, null=True)

    #> Ca H (3968.4673 +/- 200 A)
    eqw_ca_h_3968 = FloatField(null=True, help_text="Equivalent width of Ca H at 3968 A [A]")
    abs_ca_h_3968 = FloatField(null=True)
    detection_stat_ca_h_3968 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_ca_h_3968 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_ca_h_3968 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_ca_h_3968 = ArrayField(FloatField, null=True)

    #> He I (6678.151 +/- 50 A)
    eqw_he_i_6678 = FloatField(null=True, help_text="Equivalent width of He I at 6678 A [A]")
    abs_he_i_6678 = FloatField(null=True)
    detection_stat_he_i_6678 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_he_i_6678 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_he_i_6678 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_he_i_6678 = ArrayField(FloatField, null=True)

    #> He I (5875.621 +/- 50 A)
    eqw_he_i_5875 = FloatField(null=True, help_text="Equivalent width of He I at 5875 A [A]")
    abs_he_i_5875 = FloatField(null=True)
    detection_stat_he_i_5875 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_he_i_5875 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_he_i_5875 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_he_i_5875 = ArrayField(FloatField, null=True)

    #> He I (5015.678 +/- 50 A)
    eqw_he_i_5015 = FloatField(null=True, help_text="Equivalent width of He I at 5015 A [A]")
    abs_he_i_5015 = FloatField(null=True)
    detection_stat_he_i_5015 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_he_i_5015 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_he_i_5015 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_he_i_5015 = ArrayField(FloatField, null=True)

    #> He I (4471.479 +/- 50 A)
    eqw_he_i_4471 = FloatField(null=True, help_text="Equivalent width of He I at 4471 A [A]")
    abs_he_i_4471 = FloatField(null=True)
    detection_stat_he_i_4471 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_he_i_4471 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_he_i_4471 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_he_i_4471 = ArrayField(FloatField, null=True)

    #> He II (4685.7 +/- 50 A)
    eqw_he_ii_4685 = FloatField(null=True, help_text="Equivalent width of He II at 4685 A [A]")
    abs_he_ii_4685 = FloatField(null=True)
    detection_stat_he_ii_4685 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_he_ii_4685 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_he_ii_4685 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_he_ii_4685 = ArrayField(FloatField, null=True)

    #> N II (6583.45 +/- 50 A)
    eqw_n_ii_6583 = FloatField(null=True, help_text="Equivalent width of N II at 6583 A [A]")
    abs_n_ii_6583 = FloatField(null=True)
    detection_stat_n_ii_6583 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_n_ii_6583 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_n_ii_6583 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_n_ii_6583 = ArrayField(FloatField, null=True)

    #> N II (6548.05 +/- 50 A)
    eqw_n_ii_6548 = FloatField(null=True, help_text="Equivalent width of N II at 6548 A [A]")
    abs_n_ii_6548 = FloatField(null=True)
    detection_stat_n_ii_6548 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_n_ii_6548 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_n_ii_6548 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_n_ii_6548 = ArrayField(FloatField, null=True)

    #> S II (6716.44 +/- 50 A)
    eqw_s_ii_6716 = FloatField(null=True, help_text="Equivalent width of S II at 6716 A [A]")
    abs_s_ii_6716 = FloatField(null=True)
    detection_stat_s_ii_6716 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_s_ii_6716 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_s_ii_6716 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_s_ii_6716 = ArrayField(FloatField, null=True)

    #> S II (6730.816 +/- 50 A)
    eqw_s_ii_6730 = FloatField(null=True, help_text="Equivalent width of S II at 6730 A [A]")
    abs_s_ii_6730 = FloatField(null=True)
    detection_stat_s_ii_6730 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_s_ii_6730 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_s_ii_6730 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_s_ii_6730 = ArrayField(FloatField, null=True)

    #> Fe II (5018.434 +/- 50 A)
    eqw_fe_ii_5018 = FloatField(null=True, help_text="Equivalent width of Fe II at 5018 A [A]")
    abs_fe_ii_5018 = FloatField(null=True)
    detection_stat_fe_ii_5018 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_fe_ii_5018 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_fe_ii_5018 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_fe_ii_5018 = ArrayField(FloatField, null=True)

    #> Fe II (5169.03 +/- 50 A)
    eqw_fe_ii_5169 = FloatField(null=True, help_text="Equivalent width of Fe II at 5169 A [A]")
    abs_fe_ii_5169 = FloatField(null=True)
    detection_stat_fe_ii_5169 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_fe_ii_5169 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_fe_ii_5169 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_fe_ii_5169 = ArrayField(FloatField, null=True)

    #> Fe II (5197.577 +/- 50 A)
    eqw_fe_ii_5197 = FloatField(null=True, help_text="Equivalent width of Fe II at 5197 A [A]")
    abs_fe_ii_5197 = FloatField(null=True)
    detection_stat_fe_ii_5197 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_fe_ii_5197 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_fe_ii_5197 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_fe_ii_5197 = ArrayField(FloatField, null=True)

    #> Fe II (6432.68 +/- 50 A)
    eqw_fe_ii_6432 = FloatField(null=True, help_text="Equivalent width of Fe II at 6432 A [A]")
    abs_fe_ii_6432 = FloatField(null=True)
    detection_stat_fe_ii_6432 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_fe_ii_6432 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_fe_ii_6432 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_fe_ii_6432 = ArrayField(FloatField, null=True)

    #> O I (5577.339 +/- 50 A)
    eqw_o_i_5577 = FloatField(null=True, help_text="Equivalent width of O I at 5577 A[A]")
    abs_o_i_5577 = FloatField(null=True)
    detection_stat_o_i_5577 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_i_5577 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_i_5577 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_i_5577 = ArrayField(FloatField, null=True)

    #> O I (6300.304 +/- 50 A)
    eqw_o_i_6300 = FloatField(null=True, help_text="Equivalent width of O I at 6300 A [A]")
    abs_o_i_6300 = FloatField(null=True)
    detection_stat_o_i_6300 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_i_6300 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_i_6300 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_i_6300 = ArrayField(FloatField, null=True)

    #> O I (6363.777 +/- 50 A)
    eqw_o_i_6363 = FloatField(null=True, help_text="Equivalent width of O I at 6363 A[A]")
    abs_o_i_6363 = FloatField(null=True)
    detection_stat_o_i_6363 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_i_6363 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_i_6363 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_i_6363 = ArrayField(FloatField, null=True)

    #> O II (3727.42 +/- 50 A)
    eqw_o_ii_3727 = FloatField(null=True, help_text="Equivalent width of O II at 3727 A [A]")
    abs_o_ii_3727 = FloatField(null=True)
    detection_stat_o_ii_3727 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_ii_3727 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_ii_3727 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_ii_3727 = ArrayField(FloatField, null=True)

    #> O III (4958.911 +/- 50 A)
    eqw_o_iii_4959 = FloatField(null=True, help_text="Equivalent width of O III at 4959 A [A]")
    abs_o_iii_4959 = FloatField(null=True)
    detection_stat_o_iii_4959 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_iii_4959 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_iii_4959 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_iii_4959 = ArrayField(FloatField, null=True)

    #> O III (5006.843 +/- 50 A)
    eqw_o_iii_5006 = FloatField(null=True, help_text="Equivalent width of O III at 5006 A [A]")
    abs_o_iii_5006 = FloatField(null=True)
    detection_stat_o_iii_5006 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_iii_5006 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_iii_5006 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_iii_5006 = ArrayField(FloatField, null=True)

    #> O III (4363.85 +/- 50 A)
    eqw_o_iii_4363 = FloatField(null=True, help_text="Equivalent width of O III at 4363 A [A]")
    abs_o_iii_4363 = FloatField(null=True)
    detection_stat_o_iii_4363 = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_o_iii_4363 = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_o_iii_4363 = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_o_iii_4363 = ArrayField(FloatField, null=True)

    #> Li I (6707.76 +/- 50 A)
    eqw_li_i = FloatField(null=True, help_text="Equivalent width of Li I at 6707 A [A]")
    abs_li_i = FloatField(null=True)
    detection_stat_li_i = FloatField(null=True, help_text="Detection probability (+1: absorption; 0: undetected; -1: emission)")
    detection_raw_li_i = FloatField(null=True, help_text="Probability that feature is not noise (0: noise, 1: confident)")
    eqw_percentiles_li_i = ArrayField(FloatField, null=True, help_text="(16, 50, 84)th percentiles of EW [mA]")
    abs_percentiles_li_i = ArrayField(FloatField, null=True)
