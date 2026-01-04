class Config:
    raw_prefix = "raw/roboflow/v8"
    aug_prefix = "processed/roboflow/augmented"
    processed_prefix = "processed/roboflow/v1"
    
    def __init__(self, env): 
        if env == "prod": self.s3_bucket = "visight-data-yusufmoola"
        else: self.s3_bucket = "visight-data-yusufmoola"
        self.catalogue_path = f"s3://{self.s3_bucket}/configs"
        if env == "prod": self.model_key = "models/augmented-yolov8m-20251202-062742" 
        else: self.model_key = "models/augmented-yolov8m-20251202-062742" 
       