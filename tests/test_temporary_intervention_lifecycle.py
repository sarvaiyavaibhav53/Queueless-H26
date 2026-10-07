"""
PRAVAAH — Step 2D.3: Temporary Intervention Lifecycle Tests.

Comprehensive test suite verifying:
1. Idle counter temporary intervention: BC -> PT -> BC lifecycle.
2. Switching delay respected before target assignment becomes active.
3. Active duration starts only when target assignment becomes active (not at request time).
4. 15-minute intervention lasts exactly 15 active minutes.
5. 30-minute intervention lasts exactly 30 active minutes.
6. Reversion restores the donor's original assignment.
7. Busy donor finishes its in-progress citizen before switching starts.
8. Currently served token is never interrupted or discarded.
9. Waiting tokens remain in their original FIFO queues without mutation.
10. Counter eligibility is strictly enforced.
11. Counter failure during intervention terminates it safely.
12. Recovery returns the counter to its original assignment (BC -> PT -> fail -> recover -> BC).
13. Only one active intervention permitted per counter.
14. Baseline produces zero intervention events.
15. Determinism: Same snapshot + seed + candidate produces identical results (CRN).
16. 15m vs 30m interventions produce different downstream queue states and wait times.
17. State cloning/isolation (candidate simulations do not mutate snapshot/state).
18. Service completion exactly at duration expiry.
19. Service completion during reversion.
20. Counter failure during switching delay.
21. Result telemetry in SimulationResult.intervention_records.
"""

from datetime import datetime, timezone, timedelta
import math
import random
import unittest
from typing import Optional

from app.interventions.models import (
    EvaluatorConfig,
    InterventionCandidate,
    PolicyType,
)
from app.interventions.evaluator import InterventionEvaluator
from app.interventions.policies import (
    candidate_to_engine_interventions,
    validate_candidate_against_state,
)
from app.simulation.engine import (
    CounterAvailabilityEvent,
    CounterReassignIntervention,
    DiscreteEventEngine,
    InterventionPhase,
)
from app.simulation.events import EventType
from app.simulation.processes import ServiceProcess
from app.simulation.results import SimulationResult, InterventionRecord
from app.simulation.scenarios import (
    SERVICE_BC_ID,
    SERVICE_PT_ID,
    SERVICE_WC_ID,
    get_standard_municipality_counters,
    get_standard_municipality_services,
)
from app.simulation.state import (
    SimCitizen,
    SimCounter,
    SimQueue,
    SimulationState,
)
from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
)


class DeterministicFixedServiceProcess(ServiceProcess):
    """Deterministic service process with configurable fixed durations for exact math testing."""
    def __init__(self, fixed_duration: float = 5.0, remaining_duration: Optional[float] = None):
        super().__init__(rng=random.Random(42))
        self.fixed_duration = float(fixed_duration)
        self.remaining_duration = float(remaining_duration) if remaining_duration is not None else float(fixed_duration)

    def sample_duration(self, mean: float, std: float) -> float:
        return self.fixed_duration

    def sample_remaining_duration(self, mean: float, std: float, elapsed: float) -> float:
        return self.remaining_duration


class TestTemporaryInterventionLifecycle(unittest.TestCase):

    def setUp(self):
        self.t_snap = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)
        self.svc_pt = ServiceConfigSnapshot(SERVICE_PT_ID, "PT", "Property Tax", 10.0, 4.0)
        self.svc_wc = ServiceConfigSnapshot(SERVICE_WC_ID, "WC", "Water Service", 15.0, 6.0)
        self.svc_bc = ServiceConfigSnapshot(SERVICE_BC_ID, "BC", "Birth Certificate", 7.0, 2.8)

        # Standard Municipality Counters:
        # C1 (PT, eligible {PT, BC}), C2 (PT, eligible {PT, WC})
        # C3 (WC, eligible {WC, PT}), C4 (WC, eligible {WC})
        # C5 (BC, eligible {BC, PT}), C6 (BC, eligible {BC})
        self.counters_snap = {
            1: CounterSnapshot(1, 1, "C1", "ACTIVE", self.t_snap, SERVICE_PT_ID, frozenset([SERVICE_PT_ID, SERVICE_BC_ID]), None),
            2: CounterSnapshot(2, 1, "C2", "ACTIVE", self.t_snap, SERVICE_PT_ID, frozenset([SERVICE_PT_ID, SERVICE_WC_ID]), None),
            3: CounterSnapshot(3, 1, "C3", "ACTIVE", self.t_snap, SERVICE_WC_ID, frozenset([SERVICE_WC_ID, SERVICE_PT_ID]), None),
            4: CounterSnapshot(4, 1, "C4", "ACTIVE", self.t_snap, SERVICE_WC_ID, frozenset([SERVICE_WC_ID]), None),
            5: CounterSnapshot(5, 1, "C5", "ACTIVE", self.t_snap, SERVICE_BC_ID, frozenset([SERVICE_BC_ID, SERVICE_PT_ID]), None),
            6: CounterSnapshot(6, 1, "C6", "ACTIVE", self.t_snap, SERVICE_BC_ID, frozenset([SERVICE_BC_ID]), None),
        }

    def _build_state(self, initial_queues=None, serving_tokens=None) -> SimulationState:
        """Helper to build an isolated SimulationState."""
        queues = {}
        for s_id in (SERVICE_PT_ID, SERVICE_WC_ID, SERVICE_BC_ID):
            q_list = []
            if initial_queues and s_id in initial_queues:
                for tok_id, tok_num, arr_min in initial_queues[s_id]:
                    q_list.append(SimCitizen(token_id=tok_id, token_number=tok_num, service_id=s_id, arrival_time_min=arr_min))
            queues[s_id] = SimQueue(service_id=s_id, waiting_citizens=q_list)

        counters = {}
        for c_id, snap in self.counters_snap.items():
            counters[c_id] = SimCounter(
                counter_id=c_id,
                office_id=1,
                name=snap.name,
                assigned_service_id=snap.assigned_service_id,
                eligible_service_ids=snap.eligible_service_ids,
                is_active=True,
                is_busy=False,
                is_switching=False,
            )

        initial_serving = {}
        if serving_tokens:
            for c_id, (s_token, elapsed) in serving_tokens.items():
                initial_serving[c_id] = (s_token, elapsed)
                counters[c_id].is_busy = True
                counters[c_id].current_token_id = s_token.token_id
                counters[c_id].current_service_id = s_token.service_id

        return SimulationState(
            office_id=1,
            office_code="MUNI",
            snapshot_time_utc=self.t_snap,
            services={
                SERVICE_PT_ID: self.svc_pt,
                SERVICE_WC_ID: self.svc_wc,
                SERVICE_BC_ID: self.svc_bc,
            },
            queues=queues,
            counters=counters,
            initial_serving_tokens=initial_serving,
        )

    # =========================================================================
    # 1. Idle Counter Lifecycle: BC -> PT -> BC
    # =========================================================================
    def test_01_idle_counter_lifecycle_bc_to_pt_to_bc(self):
        """
        Idle Counter C5 (normally BC) requested to reassign to PT:
        duration = 15m, switching delay = 3m.
        Expected timeline:
        t=0: requested, enters switching
        t=3: target assignment active (C5 -> PT), active duration starts
        t=18: active duration expires (15m active), enters reversion switching
        t=21: restored to BC
        """
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0), (102, "PT-102", 0.0)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0)],
            }
        )
        # Deactivate C1, C2 so only C5 serves PT when active
        state.counters[1].is_active = False
        state.counters[2].is_active = False
        state.counters[6].is_active = False  # Deactivate C6 so BC waits for C5

        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=5,
            new_service_id=SERVICE_PT_ID,
            duration_min=15.0,
            switching_delay_min=3.0,
            original_service_id=SERVICE_BC_ID,
        )

        engine = DiscreteEventEngine(
            horizon_minutes=30.0,
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=5.0),
            interventions=[interv],
        )

        res = engine.run(state)

        # Verify telemetry record
        self.assertEqual(len(res.intervention_records), 1)
        rec = res.intervention_records[0]
        self.assertEqual(rec.counter_id, 5)
        self.assertEqual(rec.original_service_id, SERVICE_BC_ID)
        self.assertEqual(rec.target_service_id, SERVICE_PT_ID)
        self.assertAlmostEqual(rec.requested_time_min, 0.0)
        self.assertAlmostEqual(rec.actual_active_start_min, 3.0)
        self.assertAlmostEqual(rec.actual_active_end_min, 18.0)
        self.assertAlmostEqual(rec.actual_active_duration_min, 15.0)
        self.assertAlmostEqual(rec.actual_restored_min, 21.0)
        self.assertEqual(rec.status, "RESTORED")

        # During t=3..18, C5 served PT tokens (101 at t=3, 102 at t=8)
        self.assertIn(101, res.completed_citizen_ids)
        self.assertIn(102, res.completed_citizen_ids)

        # After t=21, C5 returned to BC and served BC token 201 (at t=21, completed at t=26)
        self.assertIn(201, res.completed_citizen_ids)

    # =========================================================================
    # 2. Switching Delay Respected Before Target Assignment
    # =========================================================================
    def test_02_switching_delay_prevents_premature_service(self):
        """During the 3-minute switching window, the counter cannot serve either queue."""
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0)],
            }
        )
        state.counters[1].is_active = False
        state.counters[2].is_active = False
        state.counters[6].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=5,
            new_service_id=SERVICE_PT_ID,
            duration_min=15.0,
            switching_delay_min=3.0,
        )

        engine = DiscreteEventEngine(
            horizon_minutes=2.5,  # Horizon terminates before switching completes (2.5 < 3.0)
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=5.0),
            interventions=[interv],
        )

        res = engine.run(state)
        # In switching window, counter could not start serving either token
        self.assertEqual(len(res.completed_citizen_ids), 0)
        self.assertEqual(res.service_metrics[SERVICE_PT_ID].customers_served, 0)
        self.assertEqual(res.service_metrics[SERVICE_BC_ID].customers_served, 0)

    # =========================================================================
    # 3. Active Duration Starts When Target Assignment Becomes Active
    # =========================================================================
    def test_03_active_duration_starts_when_target_assignment_active(self):
        """Active duration does not count the initial switching delay."""
        state = self._build_state()
        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=5,
            new_service_id=SERVICE_PT_ID,
            duration_min=10.0,
            switching_delay_min=4.0,
        )
        engine = DiscreteEventEngine(
            horizon_minutes=20.0,
            seed=42,
            interventions=[interv],
        )
        res = engine.run(state)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_start_min, 4.0)
        self.assertAlmostEqual(rec.actual_active_end_min, 14.0)
        self.assertAlmostEqual(rec.actual_active_duration_min, 10.0)
        self.assertAlmostEqual(rec.actual_restored_min, 18.0)

    # =========================================================================
    # 4 & 5. 15-Minute vs 30-Minute Interventions
    # =========================================================================
    def test_04_15m_intervention_lasts_15_active_minutes(self):
        state = self._build_state()
        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=15.0, switching_delay_min=3.0,
        )
        res = DiscreteEventEngine(horizon_minutes=40.0, interventions=[interv]).run(state)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_duration_min, 15.0)
        self.assertAlmostEqual(rec.actual_restored_min, 21.0)

    def test_05_30m_intervention_lasts_30_active_minutes(self):
        state = self._build_state()
        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=30.0, switching_delay_min=3.0,
        )
        res = DiscreteEventEngine(horizon_minutes=50.0, interventions=[interv]).run(state)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_duration_min, 30.0)
        self.assertAlmostEqual(rec.actual_restored_min, 36.0)

    # =========================================================================
    # 6. Reversion Restores Original Assignment
    # =========================================================================
    def test_06_reversion_restores_original_assignment(self):
        state = self._build_state(
            initial_queues={SERVICE_BC_ID: [(301, "BC-301", 0.0)]}
        )
        # Deactivate C6 so BC queue is only served by C5 after restoration
        state.counters[6].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=10.0, switching_delay_min=2.0, original_service_id=SERVICE_BC_ID,
        )
        res = DiscreteEventEngine(
            horizon_minutes=25.0,
            service_process=DeterministicFixedServiceProcess(fixed_duration=4.0),
            interventions=[interv],
        ).run(state)

        # Restored at 14.0 (2 + 10 + 2), BC token 301 served from 14.0 to 18.0
        self.assertIn(301, res.completed_citizen_ids)
        self.assertEqual(res.intervention_records[0].status, "RESTORED")

    # =========================================================================
    # 7 & 8. Busy Donor Finishes Current Service Before Switching (No Interruption)
    # =========================================================================
    def test_07_busy_donor_finishes_current_token_before_switching(self):
        """
        C5 is currently busy serving BC token 50.
        Service finishes at t=4.0.
        Intervention requested at t=0.
        Correct behavior:
        Token 50 completes normally at t=4.0.
        Switching delay begins at t=4.0 and runs until t=7.0.
        Target assignment (PT) becomes active at t=7.0.
        Duration of 15m expires at t=22.0.
        """
        serving_snap = ServingTokenSnapshot(
            token_id=50, token_number="BC-50", service_id=SERVICE_BC_ID, counter_id=5,
            status="SERVING", called_at_utc=self.t_snap - timedelta(minutes=4),
            serving_at_utc=self.t_snap - timedelta(minutes=3),
        )
        state = self._build_state(
            initial_queues={SERVICE_PT_ID: [(101, "PT-101", 0.0)]},
            serving_tokens={5: (serving_snap, 3.0)},
        )
        state.counters[1].is_active = False
        state.counters[2].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=5,
            new_service_id=SERVICE_PT_ID,
            duration_min=15.0,
            switching_delay_min=3.0,
        )

        engine = DiscreteEventEngine(
            horizon_minutes=35.0,
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=5.0, remaining_duration=4.0),
            interventions=[interv],
        )

        res = engine.run(state)

        # 1. In-progress token 50 completed without interruption
        self.assertIn(50, res.completed_citizen_ids)
        # 2. Telemetry shows active start at 7.0 (4.0 completion + 3.0 switching)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_start_min, 7.0)
        self.assertAlmostEqual(rec.actual_active_end_min, 22.0)
        self.assertAlmostEqual(rec.actual_active_duration_min, 15.0)
        self.assertAlmostEqual(rec.actual_restored_min, 25.0)
        self.assertEqual(rec.status, "RESTORED")

    # =========================================================================
    # 9. Waiting Tokens Remain in Original FIFO Queues
    # =========================================================================
    def test_09_waiting_tokens_remain_in_original_queues(self):
        """Reassigning C5 to PT must never transfer BC tokens to PT queue."""
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0), (202, "BC-202", 0.0)],
            }
        )
        state.counters[6].is_active = False  # Only C5 serves BC

        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=10.0, switching_delay_min=2.0,
        )

        res = DiscreteEventEngine(
            horizon_minutes=12.0,  # Horizon terminates while C5 is active at PT
            service_process=DeterministicFixedServiceProcess(fixed_duration=5.0),
            interventions=[interv],
        ).run(state)

        # BC tokens remained in BC queue and are unserved
        self.assertIn(201, res.unserved_citizen_ids)
        self.assertIn(202, res.unserved_citizen_ids)
        self.assertEqual(res.service_metrics[SERVICE_BC_ID].unserved_customers_count, 2)

    # =========================================================================
    # 10. Counter Eligibility Enforced
    # =========================================================================
    def test_10_counter_eligibility_enforced(self):
        """Counter C6 is eligible ONLY for BC; assigning to PT must be rejected."""
        state = self._build_state()
        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=6,  # C6 eligible only for BC
            new_service_id=SERVICE_PT_ID,
            duration_min=15.0,
        )
        engine = DiscreteEventEngine(horizon_minutes=30.0, interventions=[interv])
        with self.assertRaises(ValueError) as ctx:
            engine.run(state)
        self.assertIn("not eligible", str(ctx.exception))

    # =========================================================================
    # 11 & 12. Counter Failure During Intervention & Recovery to Original
    # =========================================================================
    def test_11_and_12_counter_failure_terminates_and_recovers_to_original(self):
        """
        BC -> PT -> failure -> unavailable -> recovery -> BC
        C5 assigned to PT at t=3. Fails at t=8. Recovers at t=15.
        When recovered, C5 must serve BC (its original assignment), not PT.
        """
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0), (102, "PT-102", 0.0)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0)],
            }
        )
        state.counters[1].is_active = False
        state.counters[2].is_active = False
        state.counters[6].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0,
            counter_id=5,
            new_service_id=SERVICE_PT_ID,
            duration_min=20.0,
            switching_delay_min=3.0,
        )

        avail_events = [
            CounterAvailabilityEvent(time_min=8.0, counter_id=5, is_available=False),
            CounterAvailabilityEvent(time_min=15.0, counter_id=5, is_available=True),
        ]

        engine = DiscreteEventEngine(
            horizon_minutes=25.0,
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=4.0),
            interventions=[interv],
            availability_events=avail_events,
        )

        res = engine.run(state)

        rec = res.intervention_records[0]
        self.assertEqual(rec.status, "TERMINATED")
        self.assertAlmostEqual(rec.actual_active_start_min, 3.0)
        self.assertAlmostEqual(rec.actual_active_end_min, 8.0)

        # After recovery at 15.0, C5 served BC token 201 (from 15.0 to 19.0)
        self.assertIn(201, res.completed_citizen_ids)

    # =========================================================================
    # 13. Only One Active Intervention Per Counter
    # =========================================================================
    def test_13_only_one_active_intervention_per_counter(self):
        state = self._build_state()
        interv1 = CounterReassignIntervention(time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID, duration_min=15.0)
        interv2 = CounterReassignIntervention(time_min=10.0, counter_id=5, new_service_id=SERVICE_PT_ID, duration_min=15.0)

        engine = DiscreteEventEngine(horizon_minutes=30.0, interventions=[interv1, interv2])
        with self.assertRaises(ValueError) as ctx:
            engine.run(state)
        self.assertIn("multiple scheduled interventions", str(ctx.exception))

    # =========================================================================
    # 14. Baseline Produces No Intervention Events
    # =========================================================================
    def test_14_baseline_produces_no_interventions(self):
        state = self._build_state()
        res = DiscreteEventEngine(horizon_minutes=30.0, interventions=None).run(state)
        self.assertEqual(len(res.intervention_records), 0)

    # =========================================================================
    # 15. Common Random Numbers: Determinism
    # =========================================================================
    def test_15_same_snapshot_seed_candidate_produces_identical_result(self):
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0), (102, "PT-102", 0.0)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0)],
            }
        )
        cand = InterventionCandidate("C5_PT_15", counter_id=5, target_service_id=SERVICE_PT_ID, duration_min=15.0, switching_delay_min=3.0)
        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=40.0, replications=2, base_seed=123))

        res1 = evaluator._simulate_policy(state, candidate=cand, seeds=[123, 1123])
        res2 = evaluator._simulate_policy(state, candidate=cand, seeds=[123, 1123])

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.total_waiting_person_minutes, r2.total_waiting_person_minutes)
            self.assertEqual(r1.completed_citizen_ids, r2.completed_citizen_ids)

    # =========================================================================
    # 16. 15m vs 30m Produce Different Downstream Queue States
    # =========================================================================
    def test_16_different_durations_produce_different_downstream_results(self):
        """
        Comparing 15m vs 30m interventions under high PT load and ongoing BC arrivals.
        30m provides 30m of target assistance while 15m provides 15m.
        Downstream queue metrics and waiting person-minutes must genuinely differ.
        """
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(i, f"PT-{i}", 0.0) for i in range(101, 107)],  # 6 PT tokens
                SERVICE_BC_ID: [(j, f"BC-{j}", 0.0) for j in range(201, 204)],  # 3 BC tokens
            }
        )
        state.counters[1].is_active = False  # Reduce PT baseline capacity
        state.counters[6].is_active = False  # Only C5 serves BC

        cand_15 = InterventionCandidate("C5_PT_15", counter_id=5, target_service_id=SERVICE_PT_ID, duration_min=15.0, switching_delay_min=3.0)
        cand_30 = InterventionCandidate("C5_PT_30", counter_id=5, target_service_id=SERVICE_PT_ID, duration_min=30.0, switching_delay_min=3.0)

        evaluator = InterventionEvaluator(EvaluatorConfig(
            horizon_minutes=60.0,
            replications=1,
            base_seed=42,
            arrival_rates_per_hour={SERVICE_PT_ID: 12.0, SERVICE_WC_ID: 4.0, SERVICE_BC_ID: 15.0},
        ))
        res_15 = evaluator._simulate_policy(state, candidate=cand_15, seeds=[42])[0]
        res_30 = evaluator._simulate_policy(state, candidate=cand_30, seeds=[42])[0]

        # Verify that 15m and 30m produce measurably different downstream queue wait times
        self.assertNotEqual(
            res_15.total_waiting_person_minutes,
            res_30.total_waiting_person_minutes,
            "15m and 30m interventions must yield different overall waiting person-minutes",
        )
        self.assertNotEqual(
            res_15.service_metrics[SERVICE_PT_ID].total_waiting_person_minutes,
            res_30.service_metrics[SERVICE_PT_ID].total_waiting_person_minutes,
            "15m and 30m interventions must yield different PT service waiting person-minutes",
        )
        self.assertNotEqual(
            res_15.service_metrics[SERVICE_BC_ID].total_waiting_person_minutes,
            res_30.service_metrics[SERVICE_BC_ID].total_waiting_person_minutes,
            "15m and 30m interventions must yield different BC service waiting person-minutes",
        )

    # =========================================================================
    # 17. State Isolation: Candidate Runs Do Not Mutate Snapshot
    # =========================================================================
    def test_17_candidate_simulations_do_not_mutate_original_state(self):
        state = self._build_state(
            initial_queues={SERVICE_PT_ID: [(101, "PT-101", 0.0)]}
        )
        c5_orig_assigned = state.counters[5].assigned_service_id
        q_len_orig = state.queues[SERVICE_PT_ID].length()

        interv = CounterReassignIntervention(time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID, duration_min=15.0)
        engine = DiscreteEventEngine(horizon_minutes=30.0, interventions=[interv])
        engine.run(state)

        # Original state is untouched
        self.assertEqual(state.counters[5].assigned_service_id, c5_orig_assigned)
        self.assertEqual(state.queues[SERVICE_PT_ID].length(), q_len_orig)

    # =========================================================================
    # 18. Service Completion Exactly at Duration Expiry
    # =========================================================================
    def test_18_service_completion_exactly_at_duration_expiry(self):
        """
        Target service token completes at t=18.0 exactly when duration of 15m expires.
        Counter must immediately begin reversion switching without picking up a new token.
        """
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0), (102, "PT-102", 0.0), (103, "PT-103", 0.0)],
            }
        )
        state.counters[1].is_active = False
        state.counters[2].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=15.0, switching_delay_min=3.0,
        )

        # Token 101: 3.0 to 10.5 (7.5m). Token 102: 10.5 to 18.0 (7.5m).
        # Token 102 completes at exactly t=18.0, coinciding with duration expiry!
        engine = DiscreteEventEngine(
            horizon_minutes=30.0,
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=7.5),
            interventions=[interv],
        )
        res = engine.run(state)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_start_min, 3.0)
        self.assertAlmostEqual(rec.actual_active_end_min, 18.0)
        self.assertAlmostEqual(rec.actual_restored_min, 21.0)
        # Token 103 was not served by C5 because duration expired at 18.0
        self.assertNotIn(103, res.completed_citizen_ids)

    # =========================================================================
    # 19. Busy Counter at Expiry Finishes Token First
    # =========================================================================
    def test_19_busy_counter_at_expiry_finishes_token_first(self):
        """
        Token started at t=3.0 with duration 16.0m (finishes at t=19.0).
        Duration expires at t=18.0.
        Citizen must NOT be interrupted; finishes at t=19.0.
        Reversion switching runs from t=19.0 to t=22.0.
        Restored at t=22.0.
        """
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(101, "PT-101", 0.0)],
            }
        )
        state.counters[1].is_active = False
        state.counters[2].is_active = False

        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=15.0, switching_delay_min=3.0,
        )

        # Token 101 takes 16.0m (t=3.0 to t=19.0).
        # Duration expires at t=18.0 (while token 101 is still being served).
        engine = DiscreteEventEngine(
            horizon_minutes=30.0,
            seed=42,
            service_process=DeterministicFixedServiceProcess(fixed_duration=16.0),
            interventions=[interv],
        )
        res = engine.run(state)
        self.assertIn(101, res.completed_citizen_ids)
        rec = res.intervention_records[0]
        self.assertAlmostEqual(rec.actual_active_start_min, 3.0)
        # Active period extended until token 101 finishes at 19.0
        self.assertAlmostEqual(rec.actual_active_end_min, 19.0)
        self.assertAlmostEqual(rec.actual_active_duration_min, 16.0)
        self.assertAlmostEqual(rec.actual_restored_min, 22.0)
        self.assertEqual(rec.status, "RESTORED")

    # =========================================================================
    # 20. Counter Failure During Switching Delay
    # =========================================================================
    def test_20_counter_failure_during_switching_delay(self):
        state = self._build_state()
        interv = CounterReassignIntervention(
            time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID,
            duration_min=15.0, switching_delay_min=5.0,
        )
        # Fails at t=2.0 (during switching window 0..5)
        avail = [CounterAvailabilityEvent(time_min=2.0, counter_id=5, is_available=False)]
        engine = DiscreteEventEngine(
            horizon_minutes=20.0,
            interventions=[interv],
            availability_events=avail,
        )
        res = engine.run(state)
        rec = res.intervention_records[0]
        self.assertEqual(rec.status, "TERMINATED")
        self.assertIsNone(rec.actual_active_start_min)

    # =========================================================================
    # 21. Evaluator Selects Optimal Temporary Duration
    # =========================================================================
    def test_21_evaluator_evaluates_candidate_durations(self):
        """Tests that InterventionEvaluator successfully evaluates candidates with candidate_durations."""
        state = self._build_state(
            initial_queues={
                SERVICE_PT_ID: [(i, f"PT-{i}", 0.0) for i in range(101, 108)],
                SERVICE_BC_ID: [(201, "BC-201", 0.0)],
            }
        )
        config = EvaluatorConfig(
            horizon_minutes=60.0,
            replications=2,
            base_seed=42,
            candidate_durations=(15.0, 30.0),
            switching_delay_min=3.0,
        )
        evaluator = InterventionEvaluator(config)
        res = evaluator.evaluate(state)

        self.assertIsNotNone(res.selected_candidate)
        self.assertIn(res.selected_policy, (PolicyType.OPTIMIZED, PolicyType.BASELINE, PolicyType.NAIVE))
        # Verify candidate evaluations include duration tags
        cand_ids = [c.candidate.candidate_id for c in res.candidate_evaluations if c.candidate]
        self.assertTrue(any("_D15" in cid or "_D30" in cid for cid in cand_ids))


if __name__ == "__main__":
    unittest.main()
