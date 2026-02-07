from pydantic import BaseModel

class IVFInput(BaseModel):
    age: int
    amh_ng_ml: float
    cycle_day: int
    avg_follicle_size_mm: float
    follicle_count: int
    estradiol_pg_ml: float
    progesterone_ng_ml: float
    bmi: float
    basal_lh_miu_ml: float
    afc: int
    cluster_id: int