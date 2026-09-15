import numpy as np
import pandas as pd
from pathlib import Path

class SyntheticFMCGDataGenerator:
    def __init__(self, n_weeks: int = 156, seed: int = 42) -> None:
        self.n_weeks = n_weeks
        self.seed = seed
        np.random.seed(self.seed)

    def generate_macro_covariates(self) -> pd.DataFrame:
        t = np.arange(self.n_weeks)
        baseline_demand = np.full(self.n_weeks, 1_000_000.0)

        # Trend: +0.15% per week
        trend = (1 + 0.0015) ** t

        # Seasonality: 52-week annual Fourier series
        # To peak in Q4 (t ~ 48), we set a phase shift
        phase = 2 * np.pi * 48 / 52
        a1 = np.sin(phase) * 0.2
        b1 = np.cos(phase) * 0.2
        
        # Second harmonic for sharper peaks
        phase2 = 2 * np.pi * (48 * 2) / 52
        a2 = np.sin(phase2) * 0.05
        b2 = np.cos(phase2) * 0.05

        seasonality = 1.0 + (
            a1 * np.sin(2 * np.pi * 1 * t / 52) + b1 * np.cos(2 * np.pi * 1 * t / 52) +
            a2 * np.sin(2 * np.pi * 2 * t / 52) + b2 * np.cos(2 * np.pi * 2 * t / 52)
        )

        # CPI index with a random walk trend
        cpi = np.zeros(self.n_weeks)
        cpi[0] = 100.0
        for i in range(1, self.n_weeks):
            cpi[i] = cpi[i-1] + np.random.normal(0.05, 0.2)

        # Competitor promo flag (p=0.08)
        competitor_promo_flag = np.random.binomial(1, 0.08, self.n_weeks)

        # Apply depresses: aggressive competitor promotions depress baseline sales by 12%
        baseline_sales = baseline_demand * trend * seasonality
        baseline_sales = np.where(competitor_promo_flag == 1, baseline_sales * 0.88, baseline_sales)

        return pd.DataFrame({
            "week": t,
            "baseline_demand": baseline_demand,
            "trend": trend,
            "seasonality": seasonality,
            "cpi": cpi,
            "competitor_promo_flag": competitor_promo_flag,
            "baseline_sales": baseline_sales
        })

    def generate_media_spend(self) -> pd.DataFrame:
        t = np.arange(self.n_weeks)

        # TV: Pulsed campaigns every 4 weeks (€100k-€400k)
        tv_spend = np.where(t % 4 == 0, np.random.uniform(100_000, 400_000, self.n_weeks), 0)

        # Google Search: Always-on capture (€20k-€80k)
        google_search_spend = np.random.uniform(20_000, 80_000, self.n_weeks)

        # Meta Display: Performance (€30k-€120k)
        meta_display_spend = np.random.uniform(30_000, 120_000, self.n_weeks)

        # TikTok: Flighted influencer spikes (€10k-€90k)
        tiktok_spend = np.where(np.random.rand(self.n_weeks) > 0.6, np.random.uniform(10_000, 90_000, self.n_weeks), 0)

        # YouTube: Brand video support (€15k-€70k)
        youtube_spend = np.random.uniform(15_000, 70_000, self.n_weeks)

        return pd.DataFrame({
            "tv_spend": tv_spend,
            "google_search_spend": google_search_spend,
            "meta_display_spend": meta_display_spend,
            "tiktok_spend": tiktok_spend,
            "youtube_spend": youtube_spend
        })

    def apply_adstock_and_hill(self, spend_df: pd.DataFrame) -> pd.DataFrame:
        # Params: (alpha_decay, V_max, K_half_saturation, n_shape)
        params = {
            "tv_spend": (0.7, 800_000, 200_000, 1.5),
            "google_search_spend": (0.1, 300_000, 40_000, 2.0),
            "meta_display_spend": (0.4, 500_000, 70_000, 1.8),
            "tiktok_spend": (0.2, 400_000, 50_000, 2.5),
            "youtube_spend": (0.5, 350_000, 60_000, 1.2)
        }

        inc_rev_df = pd.DataFrame()

        for col, (alpha, v_max, k, n) in params.items():
            spend = spend_df[col].values
            adstocked = np.zeros(self.n_weeks)
            adstocked[0] = spend[0]
            
            # 1. Apply geometric decay (adstock)
            for i in range(1, self.n_weeks):
                adstocked[i] = spend[i] + alpha * adstocked[i-1]

            # 2. Apply Hill Saturation function
            revenue = (v_max * (adstocked ** n)) / ((k ** n) + (adstocked ** n))
            
            out_col = col.replace("_spend", "_inc_revenue")
            inc_rev_df[out_col] = revenue

        return inc_rev_df

    def assemble_and_export(self, output_parquet_path: str) -> None:
        macro_df = self.generate_macro_covariates()
        spend_df = self.generate_media_spend()
        inc_rev_df = self.apply_adstock_and_hill(spend_df)

        final_df = pd.concat([macro_df, spend_df, inc_rev_df], axis=1)

        # Calculate total observed revenue
        total_inc = inc_rev_df.sum(axis=1)
        
        # Gaussian noise (epsilon ~ N(0, sigma^2))
        sigma = (macro_df["baseline_sales"].mean() + total_inc.mean()) * 0.05
        noise = np.random.normal(0, sigma, self.n_weeks)

        final_df["total_revenue"] = macro_df["baseline_sales"] + total_inc + noise

        out_path = Path(output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(out_path, compression="snappy")
        print(f"Exported synthetic dataset to {out_path} ({self.n_weeks} weeks, {len(final_df.columns)} columns)")

if __name__ == "__main__":
    generator = SyntheticFMCGDataGenerator()
    generator.assemble_and_export("data/gold/marketing_mix_gold.parquet")
