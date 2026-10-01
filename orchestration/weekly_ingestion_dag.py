import os
import numpy as np
import pandas as pd
from dagster import asset, Definitions, ScheduleDefinition, define_asset_job

@asset
def append_weekly_spend() -> pd.DataFrame:
    """
    Simulates querying ad platforms (Meta, Google, etc.) to fetch the latest
    week of media spend and sales outcomes.
    """
    channels = ["tv", "google_search", "meta_display", "tiktok", "youtube"]
    new_data = {"week": [pd.Timestamp.now().strftime('%Y-%m-%d')]}
    
    for ch in channels:
        # Mock weekly spend
        spend = np.random.uniform(10000, 100000)
        new_data[f"{ch}_spend"] = [spend]
        # Mock incremental revenue (assuming an ROAS between 1.0 and 3.0)
        new_data[f"{ch}_inc_revenue"] = [spend * np.random.uniform(1.0, 3.0)]
        
    return pd.DataFrame(new_data)

@asset
def update_gcs_bucket(append_weekly_spend: pd.DataFrame) -> str:
    """
    Simulates downloading the existing historical Parquet file from GCS, 
    appending the new week's data, and uploading the updated file back to GCS.
    """
    bucket_name = os.environ.get("GCS_DATA_BUCKET", "mede-platform-prod-data")
    object_name = "marketing_mix_gold.parquet"
    
    # In a full implementation, the logic would be:
    # 1. client = storage.Client()
    # 2. Download blob to local temp file
    # 3. old_df = pd.read_parquet(temp_file)
    # 4. updated_df = pd.concat([old_df, append_weekly_spend], ignore_index=True)
    # 5. updated_df.to_parquet(temp_file)
    # 6. Upload blob back to GCS
    
    return f"Simulated updating gs://{bucket_name}/{object_name} with {len(append_weekly_spend)} new rows."

# Define a job that executes all assets in this module
weekly_ingestion_job = define_asset_job("weekly_ingestion_job", selection="*")

# Define a schedule to run this job every Monday at 2:00 AM
weekly_schedule = ScheduleDefinition(
    job=weekly_ingestion_job,
    cron_schedule="0 2 * * 1", 
)

# Dagster definitions entry point
defs = Definitions(
    assets=[append_weekly_spend, update_gcs_bucket],
    schedules=[weekly_schedule],
)
