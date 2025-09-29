class Config:
    s3_bucket = "visight-data-yusufmoola"
    raw_prefix = "raw/roboflow/v8"
    processed_prefix = "processed/roboflow/v1"
    catalogue_path = f"s3://{s3_bucket}/configs"