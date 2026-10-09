"""
PRAVAAH — Candidate Intervention Evaluator & Decision Engine.

Compares Baseline (No Intervention), Naive Longest-Queue, and PRAVAAH Optimized policies
under identical stochastic conditions across multiple replications.
Guarantees:
- Common Random Numbers (CRN): identical seed sequences for all candidate evaluations.
- Real P90 waiting-time constraints enforced; infeasible candidates rejected.
- Primary objective: MINIMIZE TOTAL WAITING PERSON-MINUTES among feasible candidates.
- Honest optimizer: if baseline is superior to all interventions, selects NO_INTERVENTION.
- Pure in-memory execution; zero database or ML model dependencies.
"""

import random
from typing import Dict, List, Optional, Sequence, Union

from app.interventions.candidates import (
    build_naive_intervention,
    generate_candidate_interventions,
)
from app.interventions.metrics import (
    aggregate_replications,
)
from app.interventions.models import (
    EvaluatorConfig,
    EvaluationResult,
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)
from app.interventions.policies import (
    candidate_to_engine_interventions,
    validate_candidate_against_state,
)
from app.simulation.engine import DiscreteEventEngine
from app.simulation.processes import PoissonArrivalProcess
from app.simulation.results import SimulationResult
from app.simulation.state import SimulationState
from shared.features import SimulationSnapshot


class InterventionEvaluator:
    """
    Evaluates candidate operational interventions and recommends the optimal feasible policy.
    """
    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.config = config or EvaluatorConfig()

    def _simulate_policy(
        self,
        state: SimulationState,
        candidate: Optional[InterventionCandidate],
        seeds: Sequence[int],
    ) -> List[SimulationResult]:
        """
        Runs multiple simulation replications for a given candidate using exact seed sequence.
        """
        engine_interventions = candidate_to_engine_interventions(candidate)
        results: List[SimulationResult] = []

        for seed in seeds:
            rep_rng = random.Random(seed)
            if self.config.arrival_process_factory is not None:
                arr_proc = self.config.arrival_process_factory(state, rep_rng)
            elif self.config.arrival_rates_per_hour is not None:
                arr_proc = PoissonArrivalProcess(
                    rates_per_hour=self.config.arrival_rates_per_hour,
                    rng=rep_rng,
                )
            else:
                arr_proc = None

            if self.config.service_process_factory is not None:
                svc_proc = self.config.service_process_factory(rep_rng)
            else:
                svc_proc = None

            engine = DiscreteEventEngine(
                horizon_minutes=self.config.horizon_minutes,
                seed=seed,
                arrival_process=arr_proc,
                service_process=svc_proc,
                interventions=engine_interventions,
                availability_events=list(self.config.availability_events) if self.config.availability_events else None,
            )
            # engine.run internally deep-clones state
            res = engine.run(state)
            results.append(res)

        return results

    def evaluate(
        self,
        state_or_snapshot: Union[SimulationState, SimulationSnapshot],
        config: Optional[EvaluatorConfig] = None,
    ) -> EvaluationResult:
        """
        Executes end-to-end evaluation:
        1. Formulates candidate set (Baseline, Naive, Optimized).
        2. Simulates each policy using identical random seeds per replication.
        3. Enforces P90 waiting-time constraints.
        4. Selects optimal feasible policy minimizing total waiting person-minutes.
        """
        eval_config = config or self.config

        # 1. Normalize input to SimulationState
        if isinstance(state_or_snapshot, SimulationSnapshot):
            initial_state = SimulationState.from_snapshot(state_or_snapshot)
        elif isinstance(state_or_snapshot, SimulationState):
            initial_state = state_or_snapshot
        else:
            raise TypeError(
                f"state_or_snapshot must be SimulationState or SimulationSnapshot (got {type(state_or_snapshot)})"
            )

        service_ids = list(initial_state.services.keys())

        # 2. Replications seed sequence (identical across all candidates)
        seeds = [eval_config.base_seed + (rep * 1000) for rep in range(eval_config.replications)]

        # 3. Formulate Candidate Policies
        # A. Baseline
        baseline_results = self._simulate_policy(initial_state, candidate=None, seeds=seeds)
        baseline_eval = aggregate_replications(
            policy_type=PolicyType.BASELINE,
            candidate=None,
            results=baseline_results,
            p90_thresholds=eval_config.p90_thresholds,
            service_ids=service_ids,
        )

        # B. Naive
        naive_candidate = build_naive_intervention(initial_state)
        naive_results = self._simulate_policy(initial_state, candidate=naive_candidate, seeds=seeds)
        naive_eval = aggregate_replications(
            policy_type=PolicyType.NAIVE,
            candidate=naive_candidate,
            results=naive_results,
            p90_thresholds=eval_config.p90_thresholds,
            service_ids=service_ids,
        )

        # C. PRAVAAH Optimized Candidates
        candidates = generate_candidate_interventions(
            initial_state,
            max_candidates=eval_config.max_candidates,
            durations=eval_config.candidate_durations,
            switching_delay_min=eval_config.switching_delay_min,
        )

        candidate_evaluations: List[PolicyEvaluation] = []
        for cand in candidates:
            # Validate candidate
            validate_candidate_against_state(cand, initial_state)
            cand_results = self._simulate_policy(initial_state, candidate=cand, seeds=seeds)
            cand_eval = aggregate_replications(
                policy_type=PolicyType.OPTIMIZED,
                candidate=cand,
                results=cand_results,
                p90_thresholds=eval_config.p90_thresholds,
                service_ids=service_ids,
            )
            candidate_evaluations.append(cand_eval)

        # 4. Filter Feasible Candidates and Rank
        # A candidate is eligible only if is_feasible is True (satisfies all P90 thresholds)
        all_actionable_evals: List[PolicyEvaluation] = []
        if naive_candidate is not None:
            all_actionable_evals.append(naive_eval)
        all_actionable_evals.extend(candidate_evaluations)

        feasible_evals = [e for e in all_actionable_evals if e.is_feasible]

        # PRAVAAH Decision Logic (Section 6):
        # 1. Evaluate no action (baseline_eval).
        # 2. Evaluate legal intervention candidates.
        # 3. Apply existing P90 feasibility constraints (feasible_evals).
        # 4. Identify the feasible candidate with lowest mean waiting person-minutes.
        # PRAVAAH may recommend an action ONLY if BOTH conditions are satisfied:
        # CONDITION A — 10% IMPROVEMENT: improvement >= 0.10 vs baseline
        # CONDITION B — 70% PAIRED WIN RATE: candidate_metric < no_action_metric in >= 70% of reps
        # If either condition fails, return "No intervention required" (Baseline).
        base_wait = baseline_eval.mean_total_waiting_person_minutes
        naive_wait = naive_eval.mean_total_waiting_person_minutes

        best_candidate_eval: Optional[PolicyEvaluation] = None
        improvement = 0.0
        pct_imp = 0.0
        decision_wins = 0
        decision_win_rate = 0.0
        cond_a_passed = False
        cond_b_passed = False
        decision_notes: List[str] = []

        if not feasible_evals:
            selected_eval = baseline_eval
            decision_notes.append("No intervention candidate satisfied P90 feasibility constraints.")
        else:
            # Sort feasible candidates by total waiting person-minutes ascending
            feasible_evals.sort(
                key=lambda e: (
                    e.mean_total_waiting_person_minutes,
                    e.candidate.candidate_id if e.candidate else "",
                )
            )
            best_candidate_eval = feasible_evals[0]
            cand_wait = best_candidate_eval.mean_total_waiting_person_minutes

            # Condition A: 10% reduction in mean waiting person-minutes vs no action
            if base_wait > 1e-9:
                improvement = (base_wait - cand_wait) / base_wait
            else:
                improvement = 0.0
            pct_imp = improvement * 100.0
            cond_a_passed = (improvement >= 0.10)

            # Condition B: 70% paired win rate across decision replications
            # candidate_metric < no_action_metric counts as a win. Ties do NOT count.
            rep_count = len(baseline_eval.replication_total_waits)
            cand_waits = best_candidate_eval.replication_total_waits
            base_waits = baseline_eval.replication_total_waits
            decision_wins = sum(
                1 for c_w, b_w in zip(cand_waits, base_waits)
                if c_w < b_w
            )
            decision_win_rate = decision_wins / rep_count if rep_count > 0 else 0.0
            cond_b_passed = (decision_win_rate >= 0.70)

            if cond_a_passed and cond_b_passed:
                # Both margin conditions satisfied: recommend intervention
                selected_eval = best_candidate_eval
                decision_notes.append(
                    f"Selected feasible candidate {best_candidate_eval.candidate.candidate_id if best_candidate_eval.candidate else 'ACTION'}: "
                    f"10% margin passed ({pct_imp:.2f}% >= 10.0%), "
                    f"70% win-rate passed ({decision_wins}/{rep_count} = {decision_win_rate:.1%})."
                )
            else:
                # Fails margin or win rate: default to no action
                selected_eval = baseline_eval
                if not cond_a_passed:
                    decision_notes.append(
                        f"Failed 10% improvement margin: observed {pct_imp:.2f}% < 10.0% required."
                    )
                if not cond_b_passed:
                    decision_notes.append(
                        f"Failed 70% paired win-rate: observed {decision_wins}/{rep_count} ({decision_win_rate:.1%}) < 70.0% required."
                    )

        sel_wait = selected_eval.mean_total_waiting_person_minutes
        imp_vs_base = base_wait - sel_wait
        pct_imp_vs_base = (imp_vs_base / base_wait * 100.0) if base_wait > 1e-6 else 0.0

        imp_vs_naive = naive_wait - sel_wait
        pct_imp_vs_naive = (imp_vs_naive / naive_wait * 100.0) if naive_wait > 1e-6 else 0.0

        # 6. Structured Explanation
        rejected_list = [
            {
                "candidate_id": e.candidate.candidate_id if e.candidate else e.policy_type.value,
                "reason": e.rejection_reason,
            }
            for e in all_actionable_evals
            if not e.is_feasible
        ]

        explanation = {
            "selected_policy": selected_eval.policy_type.value,
            "selected_candidate_id": (
                selected_eval.candidate.candidate_id if selected_eval.candidate else None
            ),
            "description": (
                selected_eval.candidate.description
                if selected_eval.candidate
                else "No counter reassignment (No intervention required)"
            ),
            "baseline_waiting_person_minutes": round(base_wait, 4),
            "naive_waiting_person_minutes": round(naive_wait, 4),
            "selected_waiting_person_minutes": round(sel_wait, 4),
            "person_minutes_saved_vs_baseline": round(imp_vs_base, 4),
            "person_minutes_saved_vs_naive": round(imp_vs_naive, 4),
            "pct_saved_vs_baseline": round(pct_imp_vs_base, 2),
            "pct_saved_vs_naive": round(pct_imp_vs_naive, 2),
            "rejected_candidates_count": len(rejected_list),
            "rejected_candidates": rejected_list,
            "decision_margin": {
                "condition_a_10pct_passed": cond_a_passed,
                "relative_improvement": round(improvement, 4),
                "pct_improvement": round(pct_imp, 2),
                "condition_b_70pct_wins_passed": cond_b_passed,
                "decision_wins": decision_wins,
                "decision_win_rate": round(decision_win_rate, 4),
                "best_candidate_id": (
                    best_candidate_eval.candidate.candidate_id
                    if best_candidate_eval and best_candidate_eval.candidate
                    else None
                ),
                "decision_notes": decision_notes,
            },
            "p90_wait_by_service": {
                k: round(v, 4) for k, v in selected_eval.p90_wait_by_service.items()
            },
            "p90_thresholds": eval_config.p90_thresholds,
        }

        return EvaluationResult(
            selected_policy=selected_eval.policy_type,
            selected_candidate=selected_eval.candidate,
            is_feasible=selected_eval.is_feasible,
            total_waiting_person_minutes=sel_wait,
            improvement_vs_baseline=imp_vs_base,
            pct_improvement_vs_baseline=pct_imp_vs_base,
            improvement_vs_naive=imp_vs_naive,
            pct_improvement_vs_naive=pct_imp_vs_naive,
            baseline_evaluation=baseline_eval,
            naive_evaluation=naive_eval,
            candidate_evaluations=candidate_evaluations,
            explanation=explanation,
        )
