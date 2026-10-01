import duckdb
import numpy as np
import pandas as pd
from typing import Literal, Dict, Optional

class AdstockTransformer:
    @staticmethod
    def geometric_decay(spend: np.ndarray, alpha: float) -> np.ndarray:
        """
        x_adstocked(t) = x(t) + alpha * x_adstocked(t-1)
        """
        adstocked = np.zeros_like(spend, dtype=float)
        if len(spend) > 0:
            adstocked[0] = spend[0]
            for t in range(1, len(spend)):
                adstocked[t] = spend[t] + alpha * adstocked[t-1]
        return adstocked
    
    @staticmethod
    def weibull_pdf_adstock(spend: np.ndarray, shape: float, scale: float, max_lag: int = 12) -> np.ndarray:
        """
        Calculates weights using the Weibull probability density function and applies 1D convolution.
        """
        lags = np.arange(max_lag)
        # We add a small epsilon to avoid division by zero at t=0 when shape < 1
        t = lags + 1e-9
        
        # f(t; k, λ) = (k/λ) * (t/λ)^(k-1) * e^(-(t/λ)^k)
        weights = (shape / scale) * ((t / scale) ** (shape - 1)) * np.exp(-((t / scale) ** shape))
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Apply 1D causal convolution over the spend array
        adstocked = np.convolve(spend, weights, mode='full')[:len(spend)]
        return adstocked

    @staticmethod
    def weibull_cdf_adstock(spend: np.ndarray, shape: float, scale: float, max_lag: int = 12) -> np.ndarray:
        """
        Calculates weights using the cumulative distribution function increments and applies 1D convolution.
        """
        lags = np.arange(max_lag)
        
        # F(t; k, λ) = 1 - e^(-(t/λ)^k)
        cdf = 1 - np.exp(-((lags / scale) ** shape))
        
        # w_t = F(t) - F(t-1)
        weights = np.zeros(max_lag)
        weights[0] = cdf[0]
        for t in range(1, max_lag):
            weights[t] = cdf[t] - cdf[t-1]
            
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        adstocked = np.convolve(spend, weights, mode='full')[:len(spend)]
        return adstocked


class DuckDBDataEngine:
    def __init__(self, parquet_path: str) -> None:
        """
        Connects DuckDB in-memory session pointing to the Parquet dataset.
        """
        self.parquet_path = parquet_path
        self.conn = duckdb.connect(database=':memory:')
        
        # Enable HTTPFS for cloud object storage support
        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")
        
        # If reading from GCS, create a secret using Application Default Credentials (ADC)
        # This works automatically in Cloud Run or local environments with `gcloud auth application-default login`
        if self.parquet_path.startswith("gs://"):
            try:
                self.conn.execute("CREATE SECRET gcs_secret (TYPE GCS, PROVIDER ADC);")
            except Exception as e:
                print(f"Warning: Failed to create DuckDB GCS ADC secret: {e}")
                
        # Create a view so we can query it easily
        self.conn.execute(f"CREATE VIEW gold_data AS SELECT * FROM read_parquet('{self.parquet_path}')")
        
    def get_channel_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Returns total spend, total revenue, average historical ROAS per channel.
        """
        # In our synth data we don't have explicit date ranges, but we structure the query grouping by channels
        query = """
        SELECT 
            SUM(tv_spend) as tv_total_spend, SUM(tv_inc_revenue) as tv_total_revenue,
            SUM(google_search_spend) as google_search_total_spend, SUM(google_search_inc_revenue) as google_search_total_revenue,
            SUM(meta_display_spend) as meta_display_total_spend, SUM(meta_display_inc_revenue) as meta_display_total_revenue,
            SUM(tiktok_spend) as tiktok_total_spend, SUM(tiktok_inc_revenue) as tiktok_total_revenue,
            SUM(youtube_spend) as youtube_total_spend, SUM(youtube_inc_revenue) as youtube_total_revenue
        FROM gold_data
        """
        df = self.execute_query(query)
        
        channels = ['tv', 'google_search', 'meta_display', 'tiktok', 'youtube']
        results = []
        for ch in channels:
            spend = df[f"{ch}_total_spend"].iloc[0]
            rev = df[f"{ch}_total_revenue"].iloc[0]
            aroas = rev / spend if spend > 0 else 0
            results.append({
                "channel": ch,
                "total_spend": spend,
                "total_revenue": rev,
                "aroas": aroas
            })
            
        return pd.DataFrame(results)
        
    def compute_rolling_adstock(self, channel: str, method: Literal["geometric", "weibull_pdf", "weibull_cdf"], params: Dict[str, float]) -> pd.DataFrame:
        """
        Executes an out-of-core SQL pipeline in DuckDB to return weekly adstocked metrics.
        """
        col_name = f"{channel}_spend"
        query = f"SELECT week, {col_name} FROM gold_data ORDER BY week ASC"
        df = self.execute_query(query)
        
        spend_array = df[col_name].values
        
        if method == "geometric":
            adstocked = AdstockTransformer.geometric_decay(spend_array, alpha=params.get("alpha", 0.0))
        elif method == "weibull_pdf":
            adstocked = AdstockTransformer.weibull_pdf_adstock(spend_array, shape=params["shape"], scale=params["scale"], max_lag=int(params.get("max_lag", 12)))
        elif method == "weibull_cdf":
            adstocked = AdstockTransformer.weibull_cdf_adstock(spend_array, shape=params["shape"], scale=params["scale"], max_lag=int(params.get("max_lag", 12)))
        else:
            raise ValueError(f"Unknown adstock method: {method}")
            
        return pd.DataFrame({
            "week": df["week"],
            f"{channel}_raw_spend": spend_array,
            f"{channel}_adstocked": adstocked
        })
        
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Sanitized read-only analytical execution.
        """
        upper_query = sql_query.upper()
        if any(keyword in upper_query for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
            raise ValueError("Only read queries are allowed for the analytical data engine.")
            
        return self.conn.execute(sql_query).df()