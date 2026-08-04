from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CostComponent:
    name: str
    estimate: float
    lower: float
    upper: float
    recurring: bool


@dataclass(frozen=True)
class CostProfile:
    components: Tuple[CostComponent, ...]
    additional_adenomas: float

    def total(self) -> float:
        return sum(component.estimate for component in self.components)

    def lower(self) -> float:
        return sum(component.lower for component in self.components)

    def upper(self) -> float:
        return sum(component.upper for component in self.components)

    def cost_per_additional_adenoma(self) -> float:
        return self.total() / self.additional_adenomas


@dataclass(frozen=True)
class DiscountedCost:
    year: int
    undiscounted: float
    discounted: float
    cumulative: float


def paper_cost_profile(additional_adenomas: float = 247.3867595819) -> CostProfile:
    return CostProfile(
        components=(
            CostComponent("system_license", 48000.0, 36000.0, 72000.0, True),
            CostComponent("training", 4200.0, 3500.0, 5000.0, False),
            CostComponent("feedback_dashboard", 6000.0, 4000.0, 8000.0, True),
            CostComponent("additional_pathology", 12800.0, 8000.0, 18000.0, True),
        ),
        additional_adenomas=additional_adenomas,
    )


def annual_cost(profile: CostProfile, year: int) -> float:
    return sum(component.estimate for component in profile.components if component.recurring or year == 1)


def discounted_costs(profile: CostProfile, years: int, discount_rate: float = 0.03) -> Tuple[DiscountedCost, ...]:
    records: List[DiscountedCost] = []
    cumulative = 0.0
    for year in range(1, years + 1):
        undiscounted = annual_cost(profile, year)
        discounted = undiscounted / ((1.0 + discount_rate) ** (year - 1))
        cumulative += discounted
        records.append(DiscountedCost(year, undiscounted, discounted, cumulative))
    return tuple(records)


def cost_per_procedure(profile: CostProfile, annual_volume: float) -> float:
    if annual_volume <= 0.0:
        raise ValueError("annual volume must be positive")
    return profile.total() / annual_volume


def cost_per_adr_point(profile: CostProfile, adr_improvement: float) -> float:
    if adr_improvement <= 0.0:
        raise ValueError("ADR improvement must be positive")
    return profile.total() / adr_improvement


def incremental_adenomas(annual_volume: float, adr_improvement_pp: float) -> float:
    return annual_volume * adr_improvement_pp / 100.0


def implied_cost_per_adenoma(total_cost: float, annual_volume: float, adr_improvement_pp: float) -> float:
    additions = incremental_adenomas(annual_volume, adr_improvement_pp)
    if additions <= 0.0:
        raise ValueError("additional adenomas must be positive")
    return total_cost / additions


def budget_impact(centers: int, years: int, profile: Optional[CostProfile] = None, discount_rate: float = 0.03) -> Mapping[str, float]:
    selected = profile if profile is not None else paper_cost_profile()
    records = discounted_costs(selected, years, discount_rate)
    return {
        "centers": float(centers),
        "years": float(years),
        "undiscounted": centers * sum(record.undiscounted for record in records),
        "discounted": centers * records[-1].cumulative,
        "first_year": centers * records[0].undiscounted,
    }


def monte_carlo_cost(profile: CostProfile, repetitions: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    totals = np.zeros(repetitions, dtype=np.float64)
    for component in profile.components:
        mode_fraction = (component.estimate - component.lower) / (component.upper - component.lower)
        mode_fraction = min(max(mode_fraction, 0.0), 1.0)
        totals += generator.triangular(component.lower, component.lower + mode_fraction * (component.upper - component.lower), component.upper, repetitions)
    return totals


def probabilistic_summary(profile: CostProfile, repetitions: int = 200000, seed: int = 2025) -> Mapping[str, float]:
    totals = monte_carlo_cost(profile, repetitions, seed)
    per_adenoma = totals / profile.additional_adenomas
    return {
        "mean_total": float(np.mean(totals)),
        "median_total": float(np.median(totals)),
        "total_lower": float(np.quantile(totals, 0.025)),
        "total_upper": float(np.quantile(totals, 0.975)),
        "mean_per_adenoma": float(np.mean(per_adenoma)),
        "per_adenoma_lower": float(np.quantile(per_adenoma, 0.025)),
        "per_adenoma_upper": float(np.quantile(per_adenoma, 0.975)),
    }


def return_per_dollar(effect_pp: float, annual_cost: float) -> float:
    if annual_cost <= 0.0:
        raise ValueError("cost must be positive")
    return effect_pp / annual_cost


def implementation_priorities() -> List[Mapping[str, float]]:
    values = [
        {"component": "feedback", "effect_pp": 2.1, "annual_cost": 6000.0},
        {"component": "alert_management", "effect_pp": 1.4, "annual_cost": 0.0},
        {"component": "training", "effect_pp": 0.8, "annual_cost": 4200.0},
        {"component": "detection", "effect_pp": 1.5, "annual_cost": 48000.0},
    ]
    for record in values:
        record["effect_per_thousand"] = float("inf") if record["annual_cost"] == 0.0 else 1000.0 * record["effect_pp"] / record["annual_cost"]
    return sorted(values, key=lambda record: record["effect_per_thousand"], reverse=True)
