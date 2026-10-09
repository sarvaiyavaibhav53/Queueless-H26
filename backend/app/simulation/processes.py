"""
PRAVAAH — Simulation Stochastic Processes.

Provides service duration sampling and customer arrival generation.
Guarantees:
- Seeded, deterministic randomness encapsulated within an isolated random.Random instance.
- No global random state mutation.
- Non-negative service duration enforcement with zero-variance safety.
- Configurable arrival processes (Poisson and Deterministic).
"""

import abc
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Set, Tuple

from app.simulation.state import SimCitizen


def derive_lognormal_params(mean: float, cv: float) -> Tuple[float, float]:
    """
    Derives the underlying normal distribution parameters (mu_log, sigma_log)
    for a Lognormal distribution having the specified arithmetic mean and
    coefficient of variation (CV = std / mean).

    Mathematical derivation:
    Let X ~ Lognormal(mu_log, sigma_log^2).
      E[X]   = exp(mu_log + sigma_log^2 / 2) = mean
      Var[X] = exp(2 * mu_log + sigma_log^2) * (exp(sigma_log^2) - 1) = (cv * mean)^2
      CV^2   = Var[X] / E[X]^2 = exp(sigma_log^2) - 1

    Therefore:
      sigma_log = sqrt(ln(1 + cv^2))
      mu_log    = ln(mean) - 0.5 * sigma_log^2

    Guarantees:
    - mean > 0
    - cv >= 0
    - When cv = 0, sigma_log = 0.0, mu_log = ln(mean), exp(mu_log) = mean
    """
    if mean <= 0.0:
        raise ValueError(f"Service arithmetic mean must be > 0.0 (got {mean})")
    if cv < 0.0:
        raise ValueError(f"Coefficient of variation (cv) must be >= 0.0 (got {cv})")

    if cv <= 1e-9:
        return math.log(mean), 0.0

    sigma_log_sq = math.log(1.0 + cv * cv)
    sigma_log = math.sqrt(sigma_log_sq)
    mu_log = math.log(mean) - 0.5 * sigma_log_sq
    return mu_log, sigma_log


class ServiceProcess:
    """
    Samples service durations from configured mean and standard deviation.
    """
    def __init__(self, rng: random.Random):
        self._rng = rng

    def sample_duration(self, mean: float, std: float, min_duration: float = 0.5) -> float:
        """
        Samples a positive service duration.
        Guarantees:
        - Never produces negative service duration.
        - Returns mean if std is 0.0 or negligible.
        - Uses truncated normal sampling with fallback clamp.
        """
        if mean <= 0.0:
            raise ValueError(f"Service mean duration must be > 0.0 (got {mean})")
        if std <= 1e-6:
            return float(mean)

        min_allowed = min(min_duration, mean * 0.2)

        # Rejection sampling (up to 25 attempts)
        for _ in range(25):
            val = self._rng.gauss(mean, std)
            if val >= min_allowed:
                return float(val)

        # Safe fallback: guarantee strictly positive duration
        return float(max(min_allowed, mean))

    def sample_remaining_duration(
        self,
        mean: float,
        std: float,
        elapsed: float,
        min_remaining: float = 0.5,
    ) -> float:
        """
        Calculates remaining service duration for a token already in progress at snapshot start.
        """
        if elapsed < 0.0:
            elapsed = 0.0

        total_sample = self.sample_duration(mean, std, min_duration=min_remaining)
        if total_sample > elapsed:
            return float(total_sample - elapsed)

        # If already elapsed longer than sampled duration, customer finishes shortly
        return float(min(min_remaining, mean * 0.1))


class LognormalServiceProcess(ServiceProcess):
    """
    Samples right-skewed service durations from a Lognormal distribution
    derived from arithmetic mean and coefficient of variation (default CV = 0.40).

    Guarantees:
    - Service duration >= 0.0 (strictly non-negative).
    - Deterministic behavior for the same random seed.
    - No NaN, no infinity.
    """
    def __init__(self, rng: random.Random, default_cv: float = 0.40):
        super().__init__(rng)
        if default_cv < 0.0:
            raise ValueError(f"default_cv must be >= 0.0 (got {default_cv})")
        self.default_cv = float(default_cv)

    def sample_duration(
        self,
        mean: float,
        std: Optional[float] = None,
        cv: Optional[float] = None,
        min_duration: float = 0.1,
    ) -> float:
        if mean <= 0.0:
            raise ValueError(f"Service mean duration must be > 0.0 (got {mean})")

        effective_cv = cv
        if effective_cv is None:
            if std is not None and std > 0.0:
                effective_cv = std / mean
            else:
                effective_cv = self.default_cv

        if effective_cv <= 1e-9:
            return float(mean)

        mu_log, sigma_log = derive_lognormal_params(mean, effective_cv)
        z = self._rng.gauss(0.0, 1.0)
        val = math.exp(mu_log + z * sigma_log)

        if not math.isfinite(val) or val < min_duration:
            val = max(min_duration, val if math.isfinite(val) else mean)
        return float(val)

    def sample_remaining_duration(
        self,
        mean: float,
        std: Optional[float] = None,
        cv: Optional[float] = None,
        elapsed: float = 0.0,
        min_remaining: float = 0.1,
    ) -> float:
        if elapsed < 0.0:
            elapsed = 0.0

        total_sample = self.sample_duration(
            mean=mean, std=std, cv=cv, min_duration=min_remaining
        )
        if total_sample > elapsed:
            return float(total_sample - elapsed)
        return float(min(min_remaining, mean * 0.1))


class BaseArrivalProcess(abc.ABC):
    """
    Abstract interface for future arrival generation.
    """
    @abc.abstractmethod
    def generate_arrivals(
        self,
        horizon_minutes: float,
        start_token_id: int = 10000,
    ) -> List[SimCitizen]:
        """Generates list of future arriving citizens within [0.0, horizon_minutes]."""
        pass


class DeterministicArrivalProcess(BaseArrivalProcess):
    """
    Produces an exact, deterministic list of customer arrivals.
    Useful for predictable unit tests and replay scenarios.
    """
    def __init__(self, arrivals: List[Tuple[float, int, Optional[str]]]):
        """
        arrivals: List of (arrival_time_min, service_id, optional_token_number)
        """
        self.arrivals = arrivals

    def generate_arrivals(
        self,
        horizon_minutes: float,
        start_token_id: int = 10000,
    ) -> List[SimCitizen]:
        citizens: List[SimCitizen] = []
        token_id = start_token_id

        for entry in self.arrivals:
            arr_time = entry[0]
            service_id = entry[1]
            tok_num = entry[2] if len(entry) > 2 and entry[2] else f"SIM-{service_id}-{token_id}"

            if 0.0 <= arr_time <= horizon_minutes:
                citizens.append(
                    SimCitizen(
                        token_id=token_id,
                        token_number=tok_num,
                        service_id=service_id,
                        arrival_time_min=float(arr_time),
                    )
                )
                token_id += 1

        citizens.sort(key=lambda c: c.arrival_time_min)
        return citizens


class PoissonArrivalProcess(BaseArrivalProcess):
    """
    Generates arrivals according to homogeneous Poisson processes per service.
    Rate parameter: citizens per hour (lambda_hour).
    """
    def __init__(self, rates_per_hour: Dict[int, float], rng: random.Random):
        self.rates_per_hour = rates_per_hour
        self._rng = rng

    def generate_arrivals(
        self,
        horizon_minutes: float,
        start_token_id: int = 10000,
    ) -> List[SimCitizen]:
        citizens: List[SimCitizen] = []
        token_id = start_token_id

        for service_id, rate_per_hour in sorted(self.rates_per_hour.items()):
            if rate_per_hour <= 0.0:
                continue

            rate_per_min = rate_per_hour / 60.0
            t_curr = 0.0

            while True:
                interarrival = self._rng.expovariate(rate_per_min)
                t_curr += interarrival
                if t_curr > horizon_minutes:
                    break

                citizens.append(
                    SimCitizen(
                        token_id=token_id,
                        token_number=f"SIM-{service_id}-{token_id}",
                        service_id=service_id,
                        arrival_time_min=float(t_curr),
                    )
                )
                token_id += 1

        citizens.sort(key=lambda c: c.arrival_time_min)
        return citizens


@dataclass(frozen=True)
class ArrivalPhase:
    """
    Time phase defining arrival behavior for a specific time interval [start_min, end_min).
    """
    phase_name: str
    start_min: float
    end_min: float
    # Target traffic intensity (rho = lambda / mu) per service: service_id -> rho
    target_rho_by_service: Dict[int, float] = field(default_factory=dict)
    # Optional explicit arrival rates in citizens/hour: service_id -> lambda_hour
    arrival_rates_per_hour: Optional[Dict[int, float]] = None
    # Optional phase-wide multipliers: service_id -> multiplier
    phase_multipliers: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_min < 0.0:
            raise ValueError(f"Phase start_min must be >= 0.0 (got {self.start_min})")
        if self.end_min <= self.start_min:
            raise ValueError(
                f"Phase end_min ({self.end_min}) must be > start_min ({self.start_min})"
            )


@dataclass(frozen=True)
class ArrivalBurst:
    """
    Temporary arrival surge superimposed during [start_min, end_min) for a specific service.
    """
    burst_name: str
    service_id: int
    start_min: float
    end_min: float
    multiplier: float = 1.0               # Multiplier applied to background rate
    additional_rate_per_hour: float = 0.0  # Additive arrivals/hour
    announced: bool = False

    def __post_init__(self) -> None:
        if self.start_min < 0.0:
            raise ValueError(f"Burst start_min must be >= 0.0 (got {self.start_min})")
        if self.end_min <= self.start_min:
            raise ValueError(
                f"Burst end_min ({self.end_min}) must be > start_min ({self.start_min})"
            )
        if self.multiplier < 0.0:
            raise ValueError(f"Burst multiplier must be >= 0.0 (got {self.multiplier})")
        if self.additional_rate_per_hour < 0.0:
            raise ValueError(
                f"Burst additional_rate_per_hour must be >= 0.0 (got {self.additional_rate_per_hour})"
            )


class PhaseArrivalProcess(BaseArrivalProcess):
    """
    Piecewise-constant Poisson arrival process driven by target traffic intensities
    (rho = lambda / mu) per phase, with support for temporary arrival bursts and time-varying rates.
    """
    def __init__(
        self,
        phases: List[ArrivalPhase],
        service_capacities_per_hour: Dict[int, float],
        bursts: Optional[List[ArrivalBurst]] = None,
        rng: Optional[random.Random] = None,
    ):
        if not phases:
            raise ValueError("phases list cannot be empty")
        self.phases = sorted(phases, key=lambda p: p.start_min)
        self.service_capacities_per_hour = dict(service_capacities_per_hour)
        self.bursts = list(bursts) if bursts else []
        self._rng = rng if rng is not None else random.Random(42)

    def get_arrival_rate_per_hour(self, service_id: int, time_min: float) -> float:
        """
        Determines instantaneous arrival rate (citizens/hour) for service_id at time_min.
        """
        active_phase: Optional[ArrivalPhase] = None
        for p in self.phases:
            if p.start_min <= time_min < p.end_min:
                active_phase = p
                break
        if active_phase is None:
            active_phase = (
                self.phases[-1] if time_min >= self.phases[-1].end_min else self.phases[0]
            )

        base_rate = 0.0
        if (
            active_phase.arrival_rates_per_hour
            and service_id in active_phase.arrival_rates_per_hour
        ):
            base_rate = active_phase.arrival_rates_per_hour[service_id]
        elif service_id in active_phase.target_rho_by_service:
            target_rho = active_phase.target_rho_by_service[service_id]
            mu = self.service_capacities_per_hour.get(service_id, 0.0)
            base_rate = target_rho * mu

        phase_mult = active_phase.phase_multipliers.get(service_id, 1.0)
        rate = base_rate * phase_mult

        for b in self.bursts:
            if b.service_id == service_id and b.start_min <= time_min < b.end_min:
                rate = rate * b.multiplier + b.additional_rate_per_hour

        return max(0.0, rate)

    def generate_arrivals(
        self,
        horizon_minutes: float,
        start_token_id: int = 10000,
    ) -> List[SimCitizen]:
        """
        Generates arrivals across piecewise-constant time intervals for each service.
        """
        citizens: List[SimCitizen] = []
        token_id = start_token_id

        all_services: Set[int] = set(self.service_capacities_per_hour.keys())
        for p in self.phases:
            all_services.update(p.target_rho_by_service.keys())
            if p.arrival_rates_per_hour:
                all_services.update(p.arrival_rates_per_hour.keys())
        for b in self.bursts:
            all_services.add(b.service_id)

        for service_id in sorted(all_services):
            boundaries: Set[float] = {0.0, float(horizon_minutes)}
            for p in self.phases:
                if 0.0 <= p.start_min <= horizon_minutes:
                    boundaries.add(p.start_min)
                if 0.0 <= p.end_min <= horizon_minutes:
                    boundaries.add(p.end_min)
            for b in self.bursts:
                if b.service_id == service_id:
                    if 0.0 <= b.start_min <= horizon_minutes:
                        boundaries.add(b.start_min)
                    if 0.0 <= b.end_min <= horizon_minutes:
                        boundaries.add(b.end_min)

            intervals = sorted(boundaries)
            for i in range(len(intervals) - 1):
                t_start = intervals[i]
                t_end = intervals[i + 1]
                if t_start >= t_end:
                    continue

                rate_hr = self.get_arrival_rate_per_hour(service_id, t_start + 1e-5)
                if rate_hr <= 0.0:
                    continue

                rate_min = rate_hr / 60.0
                t_curr = t_start

                while True:
                    dt = self._rng.expovariate(rate_min)
                    t_curr += dt
                    if t_curr >= t_end:
                        break
                    citizens.append(
                        SimCitizen(
                            token_id=token_id,
                            token_number=f"SIM-{service_id}-{token_id}",
                            service_id=service_id,
                            arrival_time_min=float(t_curr),
                        )
                    )
                    token_id += 1

        citizens.sort(key=lambda c: (c.arrival_time_min, c.token_id))
        return citizens

