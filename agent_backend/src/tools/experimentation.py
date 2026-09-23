import numpy as np
from scipy import stats
from typing import Dict, Any

class ExperimentAuditor:
    @staticmethod
    def audit_sample_ratio_mismatch(
        observed_control: int, 
        observed_variant: int, 
        expected_control_ratio: float = 0.5, 
        alpha: float = 0.01
    ) -> Dict[str, Any]:
        """
        Executes Pearson's Chi-Square goodness-of-fit test for SRM.
        Detects if traffic was split incorrectly (e.g. tracking bug).
        """
        total = observed_control + observed_variant
        expected_control = total * expected_control_ratio
        expected_variant = total * (1.0 - expected_control_ratio)
        
        observed = [observed_control, observed_variant]
        expected = [expected_control, expected_variant]
        
        # Pearson's Chi-Square test
        chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        
        srm_detected = bool(p_value < alpha)
        
        if srm_detected:
            if p_value < 0.001:
                severity = "Critical"
            else:
                severity = "High"
        else:
            severity = "None"
            
        return {
            "srm_detected": srm_detected,
            "p_value": float(p_value),
            "chi2_stat": float(chi2_stat),
            "severity": severity
        }

    @staticmethod
    def calculate_statistical_power(
        sample_size_per_arm: int, 
        baseline_p: float, 
        relative_lift: float, 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Computes power for two-sample proportion Z-test.
        Evaluates if the experiment actually had enough data to detect the lift.
        """
        p1 = baseline_p
        p2 = baseline_p * (1.0 + relative_lift)
        
        # Pooled proportion
        p_pool = (p1 + p2) / 2.0
        
        # Effect size (absolute difference)
        delta = abs(p1 - p2)
        
        # Z critical value
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # Standard errors
        se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / sample_size_per_arm)
        se_ind = np.sqrt((p1 * (1 - p1) / sample_size_per_arm) + (p2 * (1 - p2) / sample_size_per_arm))
        
        # Power calculation
        z_power = (delta - z_crit * se_pool) / se_ind
        power = float(stats.norm.cdf(z_power))
        
        is_underpowered = power < 0.80
        
        return {
            "power": power,
            "is_underpowered": is_underpowered,
            "baseline_p": p1,
            "variant_p": p2,
            "delta": delta
        }

    @staticmethod
    def calculate_required_sample_size(
        baseline_p: float, 
        min_relative_lift: float, 
        target_power: float = 0.80, 
        alpha: float = 0.05
    ) -> int:
        """
        Returns minimum sample size per variant arm to achieve target power.
        Using standard normal approximation for proportions.
        """
        p1 = baseline_p
        p2 = baseline_p * (1.0 + min_relative_lift)
        p_pool = (p1 + p2) / 2.0
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(target_power)
        
        # Standard errors squared term
        num = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        den = (p1 - p2) ** 2
        
        n_per_arm = int(np.ceil(num / den))
        return n_per_arm

class GeoExperimentAuditor:
    @staticmethod
    def simulate_null_drift(
        historical_control: np.ndarray,
        historical_treatment: np.ndarray,
        test_duration_days: int,
        n_bootstraps: int = 2000,
        random_seed: int = 42
    ) -> np.ndarray:
        """
        Simulates the distribution of drift between two groups under H0 
        using block-bootstrap sampling of historical pre-period data.
        """
        np.random.seed(random_seed)
        n_days = len(historical_control)
        max_start = n_days - test_duration_days
        
        if max_start <= 0:
            raise ValueError("Historical data must be longer than test duration.")
            
        drift_distribution = []
        
        for _ in range(n_bootstraps):
            # Sample a random historical window of length `test_duration_days`
            idx = np.random.randint(0, max_start)
            c_slice = historical_control[idx:idx + test_duration_days]
            t_slice = historical_treatment[idx:idx + test_duration_days]
            
            # Scaled counterfactual prediction (ratio estimator under H0)
            baseline_ratio = np.sum(historical_control[:idx]) / np.sum(historical_treatment[:idx]) if idx > 14 else np.sum(c_slice) / (np.sum(t_slice) + 1e-9)
            
            predicted_treatment = np.sum(c_slice) / (baseline_ratio + 1e-9)
            actual_treatment = np.sum(t_slice)
            
            # Empirical percentage drift under H0
            pct_drift = (actual_treatment - predicted_treatment) / (predicted_treatment + 1e-9)
            drift_distribution.append(pct_drift)
            
        return np.array(drift_distribution)

    @staticmethod
    def compute_empirical_mde(
        drift_distribution: np.ndarray,
        alpha: float = 0.05,
        target_power: float = 0.80
    ) -> Dict[str, Any]:
        """
        Computes the Minimum Detectable Effect (MDE) directly from the empirical H0 distribution.
        """
        # Critical boundaries for false positive control under empirical H0
        lower_crit = np.percentile(drift_distribution, (alpha / 2.0) * 100)
        upper_crit = np.percentile(drift_distribution, (1.0 - alpha / 2.0) * 100)
        
        # Standard error of the counterfactual prediction
        std_error = np.std(drift_distribution)
        
        # Empirical MDE (two-tailed): Shift required so that (1 - beta)% of the shifted distribution
        # clears the critical value of the null distribution
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(target_power)
        
        mde_relative = (z_alpha + z_beta) * std_error

        return {
            "empirical_std_error": float(std_error),
            "mde_relative": float(mde_relative),
            "two_tailed_ci_95": (float(lower_crit), float(upper_crit)),
            "is_sub_two_percent": bool(mde_relative < 0.02)
        }
