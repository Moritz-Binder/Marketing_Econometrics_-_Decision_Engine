import os
import numpy as np
import pytest
from src.tools.data_engine import AdstockTransformer, DuckDBDataEngine

def test_geometric_decay():
    spend = np.array([100, 0, 0, 0])
    alpha = 0.5
    adstocked = AdstockTransformer.geometric_decay(spend, alpha)
    
    # 100 -> 50 -> 25 -> 12.5
    assert adstocked[0] == 100
    assert adstocked[1] == 50
    assert adstocked[2] == 25
    assert adstocked[3] == 12.5

def test_weibull_pdf_normalization():
    spend = np.array([100, 0, 0, 0, 0])
    adstocked = AdstockTransformer.weibull_pdf_adstock(spend, shape=1.5, scale=2.0, max_lag=5)
    
    # Since weights sum to 1, the total distributed adstock of a single 100 pulse should equal 100 
    # (accounting for tiny truncation differences beyond max_lag, it will be exactly 100 in our window since mode='full')
    assert np.isclose(adstocked.sum(), 100.0, rtol=0.1)

def test_weibull_cdf_normalization():
    spend = np.array([100, 0, 0, 0, 0])
    adstocked = AdstockTransformer.weibull_cdf_adstock(spend, shape=1.5, scale=2.0, max_lag=5)
    assert np.isclose(adstocked.sum(), 100.0, rtol=0.1)
    
def test_duckdb_engine_reads_parquet():
    parquet_path = "data/gold/marketing_mix_gold.parquet"
    if not os.path.exists(parquet_path):
        pytest.skip("Synthetic dataset not generated yet.")
        
    engine = DuckDBDataEngine(parquet_path)
    
    # 1. Test Query Execution limits
    with pytest.raises(ValueError, match="Only read queries"):
        engine.execute_query("DROP TABLE gold_data")
        
    # 2. Test Channel Summary
    summary_df = engine.get_channel_summary()
    assert len(summary_df) == 5
    assert "channel" in summary_df.columns
    assert "aroas" in summary_df.columns
    
    # 3. Test Rolling Adstock output
    adstocked_df = engine.compute_rolling_adstock("tv", "geometric", {"alpha": 0.5})
    assert len(adstocked_df) == 156
    assert "tv_adstocked" in adstocked_df.columns
