import pandas as pd
from scripts.generate_synthetic_data import SyntheticFMCGDataGenerator

def test_data_generation_shapes():
    generator = SyntheticFMCGDataGenerator(n_weeks=10)
    
    macro_df = generator.generate_macro_covariates()
    assert len(macro_df) == 10
    assert "baseline_sales" in macro_df.columns
    
    spend_df = generator.generate_media_spend()
    assert len(spend_df) == 10
    assert "tv_spend" in spend_df.columns
    
    inc_rev_df = generator.apply_adstock_and_hill(spend_df)
    assert len(inc_rev_df) == 10
    assert "tv_inc_revenue" in inc_rev_df.columns
