from astra.database.sdss5db import ReflectBaseModel

class CatalogBaseModel(ReflectBaseModel):
    class Meta:
        use_reflection = True
        schema = "catalogdb"
    
class Catalog(CatalogBaseModel):
    class Meta:
        table_name = "catalog"

class CatalogToSDSSDR16ApogeeStar(CatalogBaseModel):
    class Meta:
        table_name = "catalog_to_sdss_dr16_apogeestar"

class CatalogToTICV8(CatalogBaseModel):
    class Meta:
        table_name = "catalog_to_tic_v8"

class GaiaDR2Source(CatalogBaseModel):
    class Meta:
        table_name = "gaia_dr2_source"

class SDSSApogeeAllStarMergeR13(CatalogBaseModel):
    class Meta:
        table_name = "sdss_apogeeallstarmerge_r13"

class SDSSDR16ApogeeStar(CatalogBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeestar"

class SDSSDR16ApogeeStarVisit(CatalogBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeestarvisit"

class SDSSDR16ApogeeVisit(CatalogBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeevisit"

class SDSSVBossSpall(CatalogBaseModel):
    class Meta:
        table_name = "sdssv_boss_spall"
    
class TICV8(CatalogBaseModel):
    class Meta:
        table_name = "tic_v8"