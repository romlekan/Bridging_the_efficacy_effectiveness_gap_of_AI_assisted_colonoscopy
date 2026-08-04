from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Reach:
    activation_rate: float
    patient_coverage: float

    def score(self) -> float:
        return (self.activation_rate + self.patient_coverage) / 2.0


@dataclass(frozen=True)
class Effectiveness:
    adr_improvement: float
    apc_improvement: float
    benchmark: float = 8.1

    def recovery(self) -> float:
        return 100.0 * self.adr_improvement / self.benchmark


@dataclass(frozen=True)
class Adoption:
    provider_adoption: float
    regular_use: float

    def score(self) -> float:
        return (self.provider_adoption + self.regular_use) / 2.0


@dataclass(frozen=True)
class Implementation:
    composite_fidelity: float
    training_completion: float
    feedback_access: float
    alert_adherence: float

    def component_mean(self) -> float:
        return (self.training_completion + self.feedback_access + self.alert_adherence) / 3.0

    def weakest_component(self) -> Tuple[str, float]:
        values = {
            "training_completion": self.training_completion,
            "feedback_access": self.feedback_access,
            "alert_adherence": self.alert_adherence,
        }
        return min(values.items(), key=lambda item: item[1])


@dataclass(frozen=True)
class Maintenance:
    initial_benefit: float
    month_12_benefit: float
    month_18_benefit: float

    def retention_12(self) -> float:
        return 100.0 * self.month_12_benefit / self.initial_benefit

    def retention_18(self) -> float:
        return 100.0 * self.month_18_benefit / self.initial_benefit


@dataclass(frozen=True)
class ReaimProfile:
    reach: Reach
    effectiveness: Effectiveness
    adoption: Adoption
    implementation: Implementation
    maintenance: Maintenance

    def normalized_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.reach.score(),
                self.effectiveness.recovery(),
                self.adoption.score(),
                self.implementation.composite_fidelity,
                self.maintenance.retention_12(),
            ],
            dtype=np.float64,
        )

    def composite(self, weights: Optional[Sequence[float]] = None) -> float:
        values = self.normalized_vector()
        if weights is None:
            return float(np.mean(values))
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.size != values.size:
            raise ValueError("weights differ from dimensions")
        return float(np.average(values, weights=weight_array))

    def mapping(self) -> Mapping[str, Mapping[str, float]]:
        return {
            "reach": {
                "activation_rate": self.reach.activation_rate,
                "patient_coverage": self.reach.patient_coverage,
                "score": self.reach.score(),
            },
            "effectiveness": {
                "adr_improvement": self.effectiveness.adr_improvement,
                "apc_improvement": self.effectiveness.apc_improvement,
                "recovery": self.effectiveness.recovery(),
            },
            "adoption": {
                "provider_adoption": self.adoption.provider_adoption,
                "regular_use": self.adoption.regular_use,
                "score": self.adoption.score(),
            },
            "implementation": {
                "composite_fidelity": self.implementation.composite_fidelity,
                "training_completion": self.implementation.training_completion,
                "feedback_access": self.implementation.feedback_access,
                "alert_adherence": self.implementation.alert_adherence,
            },
            "maintenance": {
                "month_12_benefit": self.maintenance.month_12_benefit,
                "month_18_benefit": self.maintenance.month_18_benefit,
                "retention_12": self.maintenance.retention_12(),
                "retention_18": self.maintenance.retention_18(),
            },
        }


def paper_profile() -> ReaimProfile:
    return ReaimProfile(
        Reach(89.2, 97.4),
        Effectiveness(5.8, 0.22),
        Adoption(93.7, 84.3),
        Implementation(78.4, 100.0, 71.2, 64.1),
        Maintenance(5.8, 4.5, 3.8),
    )


def readiness(profile: ReaimProfile, minimum: float = 72.0) -> Mapping[str, object]:
    weakest_name, weakest_value = profile.implementation.weakest_component()
    return {
        "ready": profile.implementation.composite_fidelity >= minimum,
        "composite_fidelity": profile.implementation.composite_fidelity,
        "minimum": minimum,
        "margin": profile.implementation.composite_fidelity - minimum,
        "weakest_component": weakest_name,
        "weakest_value": weakest_value,
    }


def profile_distance(first: ReaimProfile, second: ReaimProfile) -> float:
    difference = first.normalized_vector() - second.normalized_vector()
    return float(np.sqrt(np.sum(difference * difference)))


def aggregate_profiles(profiles: Sequence[ReaimProfile], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    matrix = np.vstack([profile.normalized_vector() for profile in profiles])
    if weights is None:
        return np.mean(matrix, axis=0)
    return np.average(matrix, axis=0, weights=np.asarray(weights, dtype=np.float64))


def implementation_gaps(profile: ReaimProfile) -> Mapping[str, float]:
    return {
        "training": 100.0 - profile.implementation.training_completion,
        "feedback": 100.0 - profile.implementation.feedback_access,
        "alert_management": 100.0 - profile.implementation.alert_adherence,
        "composite": 100.0 - profile.implementation.composite_fidelity,
    }


def rank_gaps(profile: ReaimProfile) -> List[Tuple[str, float]]:
    return sorted(implementation_gaps(profile).items(), key=lambda item: item[1], reverse=True)


def projected_profile(profile: ReaimProfile, training: Optional[float] = None, feedback: Optional[float] = None, alert: Optional[float] = None) -> ReaimProfile:
    implementation = Implementation(
        composite_fidelity=np.mean(
            [
                profile.implementation.training_completion if training is None else training,
                profile.implementation.feedback_access if feedback is None else feedback,
                profile.implementation.alert_adherence if alert is None else alert,
            ]
        ),
        training_completion=profile.implementation.training_completion if training is None else training,
        feedback_access=profile.implementation.feedback_access if feedback is None else feedback,
        alert_adherence=profile.implementation.alert_adherence if alert is None else alert,
    )
    return ReaimProfile(profile.reach, profile.effectiveness, profile.adoption, implementation, profile.maintenance)
