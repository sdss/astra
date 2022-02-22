from astra.database.sdss5db import ReflectBaseModel

class CatalogDBBaseModel(ReflectBaseModel):
    class Meta:
        use_reflection = True
        schema = "catalogdb"
    
class Catalog(CatalogDBBaseModel):
    class Meta:
        table_name = "catalog"

class CatalogToSDSSDR16ApogeeStar(CatalogDBBaseModel):
    class Meta:
        table_name = "catalog_to_sdss_dr16_apogeestar"

class CatalogToTICV8(CatalogDBBaseModel):
    class Meta:
        table_name = "catalog_to_tic_v8"

class GaiaDR2Source(CatalogDBBaseModel):
    class Meta:
        table_name = "gaia_dr2_source"

class SDSSApogeeAllStarMergeR13(CatalogDBBaseModel):
    class Meta:
        table_name = "sdss_apogeeallstarmerge_r13"

class SDSSDR16ApogeeStar(CatalogDBBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeestar"

class SDSSDR16ApogeeStarVisit(CatalogDBBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeestarvisit"

class SDSSDR16ApogeeVisit(CatalogDBBaseModel):
    class Meta:
        table_name = "sdss_dr16_apogeevisit"

class SDSSVBossSpall(CatalogDBBaseModel):
    class Meta:
        table_name = "sdssv_boss_spall"
    
class TICV8(CatalogDBBaseModel):
    class Meta:
        table_name = "tic_v8"