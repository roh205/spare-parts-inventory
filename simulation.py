"""
Stochastic Spare Parts Inventory Optimization Engine
=====================================================
Formulation:
- Fleet of N identical machines subject to component wear-out.
- Component time-to-failure follows Weibull(k, lambda) renewal process.
- Single-period / Monthly planning horizon of H hours.
- Base-stock inventory policy S in [0, B].
- Costs:
    * Regular procurement: Cp ($/unit)
    * Monthly holding: Ch ($/unit/month)
    * Emergency procurement: Ce ($/unit)
    * Equipment downtime: Cd ($/stockout incident)
    * Stockout penalty: Cs = Ce + Cd
- Stochastic metrics computed:
    * Expected Total Cost E[Cost(S)]
    * Value at Risk (VaR 95%) and Conditional Value at Risk (CVaR 95%)
    * Cycle Service Level (CSL) = P(D <= S)
    * Fill Rate = 1 - E[max(0, D - S)] / E[D]
    * Value of Stochastic Solution (VSS) = Cost(S_det) - Cost(S*)
    * Expected Value of Perfect Information (EVPI)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.special as sp


@dataclass
class InventoryConfig:
    num_machines: int = 25
    hours_per_month: float = 600.0
    weibull_k: float = 2.2         # Shape parameter (k > 1 implies aging/wear-out)
    weibull_lambda: float = 1800.0 # Scale parameter in hours
    
    cost_procure: float = 1500.0   # Cp ($/unit)
    cost_holding: float = 120.0    # Ch ($/unit/month)
    cost_emergency: float = 4200.0 # Ce ($/unit)
    cost_downtime: float = 6000.0  # Cd ($/incident)
    
    max_storage: int = 40          # Upper bound for inventory S
    num_scenarios: int = 20000     # Monte Carlo sample size
    random_seed: Optional[int] = 42

    @property
    def penalty_stockout(self) -> float:
        return self.cost_emergency + self.cost_downtime

    @property
    def mtbf(self) -> float:
        """Mean Time Between Failures: lambda * Gamma(1 + 1/k)"""
        return self.weibull_lambda * float(sp.gamma(1.0 + 1.0 / self.weibull_k))

    @property
    def ttf_std(self) -> float:
        """Standard deviation of TTF"""
        g1 = float(sp.gamma(1.0 + 1.0 / self.weibull_k))
        g2 = float(sp.gamma(1.0 + 2.0 / self.weibull_k))
        var = (self.weibull_lambda ** 2) * max(0.0, g2 - g1 ** 2)
        return float(np.sqrt(var))

    @property
    def critical_fractile(self) -> float:
        """
        Newsvendor critical fractile ratio: Cu / (Cu + Co)
        Cu (underage cost) = Ce + Cd - Cp (net extra cost of an emergency stockout)
        Co (overage cost) = Ch (holding cost per unsold unit)
        """
        cu = self.penalty_stockout - self.cost_procure
        co = self.cost_holding
        return cu / (cu + co)


def simulate_breakdowns_vectorized(
    num_machines: int,
    hours: float,
    k: float,
    lam: float,
    scenarios: int,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simulates component failure renewal processes across N machines for M scenarios.
    High-performance vectorized implementation replacing nested scalar loops.
    """
    rng = np.random.default_rng(seed)
    total_failures = np.zeros(scenarios, dtype=np.int32)
    
    # Current cumulative operational time for each machine across all scenarios
    curr_time = rng.weibull(k, size=(scenarios, num_machines)) * lam
    
    while True:
        failed = curr_time <= hours
        if not np.any(failed):
            break
        # Count failures in this renewal cycle
        total_failures += np.sum(failed, axis=1)
        # Sample next time-to-failure for components that failed
        num_failed = int(np.count_nonzero(failed))
        curr_time[failed] += rng.weibull(k, size=num_failed) * lam
        # Freeze machines that did not fail within the horizon
        curr_time[~failed] = np.inf
        
    return total_failures


def evaluate_policy_cost(
    S: int,
    demands: np.ndarray,
    cost_procure: float,
    cost_holding: float,
    penalty_stockout: float
) -> Dict[str, np.ndarray]:
    """
    Computes individual cost components and total cost array for a given base-stock level S.
    """
    holding = cost_holding * np.maximum(0, S - demands)
    penalty = penalty_stockout * np.maximum(0, demands - S)
    procure = np.full_like(demands, cost_procure * S, dtype=np.float64)
    total = procure + holding + penalty
    return {
        "procure": procure,
        "holding": holding,
        "penalty": penalty,
        "total": total
    }


def compute_policy_metrics(
    S: int,
    demands: np.ndarray,
    cost_procure: float,
    cost_holding: float,
    penalty_stockout: float
) -> Dict[str, float]:
    """
    Calculates statistical, financial, and operational KPIs for a specific stock level S.
    """
    cost_dict = evaluate_policy_cost(S, demands, cost_procure, cost_holding, penalty_stockout)
    total_costs = cost_dict["total"]
    
    unmet_demand = np.maximum(0, demands - S)
    overstock = np.maximum(0, S - demands)
    stockouts_occurred = demands > S
    
    expected_cost = float(np.mean(total_costs))
    cost_std = float(np.std(total_costs))
    var95 = float(np.percentile(total_costs, 95))
    cvar95 = float(np.mean(total_costs[total_costs >= var95]))
    
    mean_demand = float(np.mean(demands))
    fill_rate = 1.0 - (float(np.mean(unmet_demand)) / (mean_demand + 1e-9))
    fill_rate = float(np.clip(fill_rate, 0.0, 1.0))
    
    csl = float(np.mean(~stockouts_occurred))  # Cycle Service Level: P(D <= S)
    stockout_prob = float(np.mean(stockouts_occurred))
    
    return {
        "stock_level": S,
        "expected_cost": expected_cost,
        "cost_std": cost_std,
        "var95": var95,
        "cvar95": cvar95,
        "mean_procure_cost": float(np.mean(cost_dict["procure"])),
        "mean_holding_cost": float(np.mean(cost_dict["holding"])),
        "mean_penalty_cost": float(np.mean(cost_dict["penalty"])),
        "expected_stockout_units": float(np.mean(unmet_demand)),
        "expected_overstock_units": float(np.mean(overstock)),
        "csl": csl,
        "fill_rate": fill_rate,
        "stockout_prob": stockout_prob
    }


def run_optimization(config: InventoryConfig) -> Dict[str, Any]:
    """
    Executes full Monte Carlo simulation, evaluates all candidate stock levels S,
    identifies S* (stochastic) vs S_det (deterministic), and computes VSS and EVPI.
    """
    demands = simulate_breakdowns_vectorized(
        num_machines=config.num_machines,
        hours=config.hours_per_month,
        k=config.weibull_k,
        lam=config.weibull_lambda,
        scenarios=config.num_scenarios,
        seed=config.random_seed
    )
    
    mean_demand = float(np.mean(demands))
    median_demand = float(np.median(demands))
    std_demand = float(np.std(demands))
    p90_demand = float(np.percentile(demands, 90))
    p95_demand = float(np.percentile(demands, 95))
    p99_demand = float(np.percentile(demands, 99))
    min_demand = int(np.min(demands))
    max_demand = int(np.max(demands))
    
    candidate_S = list(range(0, config.max_storage + 1))
    curve_data: List[Dict[str, Any]] = []
    
    for s in candidate_S:
        metrics = compute_policy_metrics(
            S=s,
            demands=demands,
            cost_procure=config.cost_procure,
            cost_holding=config.cost_holding,
            penalty_stockout=config.penalty_stockout
        )
        curve_data.append(metrics)
        
    expected_costs = np.array([m["expected_cost"] for m in curve_data])
    optimal_idx = int(np.argmin(expected_costs))
    optimal_s = candidate_S[optimal_idx]
    stochastic_metrics = curve_data[optimal_idx]
    
    # Deterministic policy: order rounded expected demand
    deterministic_s = int(np.round(mean_demand))
    # Bound within candidate range
    deterministic_s = max(0, min(config.max_storage, deterministic_s))
    deterministic_metrics = compute_policy_metrics(
        S=deterministic_s,
        demands=demands,
        cost_procure=config.cost_procure,
        cost_holding=config.cost_holding,
        penalty_stockout=config.penalty_stockout
    )
    
    # Value of Stochastic Solution (VSS)
    vss = deterministic_metrics["expected_cost"] - stochastic_metrics["expected_cost"]
    vss_pct = (vss / (deterministic_metrics["expected_cost"] + 1e-9)) * 100.0
    
    # Wait-and-See (WS) cost for Expected Value of Perfect Information (EVPI)
    # Perfect information: for each scenario m, we know demand D[m] in advance, so S = D[m]
    # In that case, Holding = 0, Penalty = 0, Cost = Cp * D[m]
    ws_costs = config.cost_procure * demands
    expected_ws_cost = float(np.mean(ws_costs))
    evpi = stochastic_metrics["expected_cost"] - expected_ws_cost
    
    # Demand histogram distribution
    unique_vals, counts = np.unique(demands, return_counts=True)
    demand_dist = [
        {"demand": int(val), "count": int(cnt), "prob": float(cnt / config.num_scenarios)}
        for val, cnt in zip(unique_vals, counts)
    ]
    
    # Realized cost samples for optimal vs deterministic (for density/histogram)
    cost_samples_opt = evaluate_policy_cost(
        optimal_s, demands, config.cost_procure, config.cost_holding, config.penalty_stockout
    )["total"]
    cost_samples_det = evaluate_policy_cost(
        deterministic_s, demands, config.cost_procure, config.cost_holding, config.penalty_stockout
    )["total"]
    
    return {
        "config": asdict(config),
        "equipment": {
            "mtbf_hours": config.mtbf,
            "ttf_std_hours": config.ttf_std,
            "critical_fractile": config.critical_fractile
        },
        "demand_stats": {
            "mean": mean_demand,
            "median": median_demand,
            "std": std_demand,
            "p90": p90_demand,
            "p95": p95_demand,
            "p99": p99_demand,
            "min": min_demand,
            "max": max_demand,
            "distribution": demand_dist
        },
        "optimal_policy": {
            "S_stochastic": optimal_s,
            "metrics": stochastic_metrics
        },
        "deterministic_policy": {
            "S_deterministic": deterministic_s,
            "metrics": deterministic_metrics
        },
        "vss": {
            "value": float(vss),
            "percentage": float(vss_pct)
        },
        "evpi": {
            "value": float(evpi),
            "ws_cost": float(expected_ws_cost)
        },
        "curve_data": curve_data,
        "cost_samples": {
            "stochastic": [float(x) for x in cost_samples_opt[:1000]],
            "deterministic": [float(x) for x in cost_samples_det[:1000]]
        }
    }


def run_sensitivity_analysis(
    config: InventoryConfig,
    k_values: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates optimal base-stock policy and costs across different Weibull wear rates (k).
    """
    if k_values is None:
        k_values = [1.2, 1.6, 2.0, 2.2, 2.5, 3.0]
        
    results = []
    candidate_S = list(range(0, config.max_storage + 1))
    
    for k_val in k_values:
        d_test = simulate_breakdowns_vectorized(
            num_machines=config.num_machines,
            hours=config.hours_per_month,
            k=k_val,
            lam=config.weibull_lambda,
            scenarios=min(config.num_scenarios, 10000),
            seed=config.random_seed
        )
        
        costs_profile = [
            float(np.mean(evaluate_policy_cost(
                s, d_test, config.cost_procure, config.cost_holding, config.penalty_stockout
            )["total"]))
            for s in candidate_S
        ]
        
        best_idx = int(np.argmin(costs_profile))
        best_s = candidate_S[best_idx]
        min_cost = costs_profile[best_idx]
        
        mean_d = float(np.mean(d_test))
        det_s = int(np.round(mean_d))
        det_cost = float(np.mean(evaluate_policy_cost(
            det_s, d_test, config.cost_procure, config.cost_holding, config.penalty_stockout
        )["total"]))
        
        mtbf = config.weibull_lambda * float(sp.gamma(1.0 + 1.0 / k_val))
        
        results.append({
            "k_shape": k_val,
            "mtbf_hours": float(mtbf),
            "mean_demand": mean_d,
            "optimal_s": best_s,
            "stochastic_cost": min_cost,
            "deterministic_s": det_s,
            "deterministic_cost": det_cost,
            "vss": det_cost - min_cost
        })
        
    return results

