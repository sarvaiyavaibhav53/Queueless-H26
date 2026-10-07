"""
PRAVAAH — Step 2C.3: Discrete-Event Simulation Engine Unit & Integration Tests.

Verifies:
A. Empty system completes safely.
B. Existing waiting queue preserves FIFO and is served by available eligible counter.
C. Multiple services maintain independent queues and eligible assignments.
D. Existing serving token respects elapsed service and completes at correct future time.
E. Service duration sampling produces reproducible, non-negative values.
F. Arrival process reproducibility and seed differentiation.
G. Waiting person-minutes verified against manually calculated exact scenario.
H. Counter unavailable prevents new service.
I. Counter recovery allows resumption of service.
J. Horizon termination strictly blocks events beyond horizon.
K. Deep clone provides total isolation between candidate runs.
L. Determinism: same state + seed produces bitwise identical results.
M. Multi-counter parallel dispatch.
N. Future intervention hook allows scheduled counter reassignment with eligibility validation.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure project root and backend are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.simulation import (
    CounterAvailabilityEvent,
    CounterReassignIntervention,
    DeterministicArrivalProcess,
    DiscreteEventEngine,
    PoissonArrivalProcess,
    ServiceProcess,
    SimCitizen,
    SimCounter,
    SimQueue,
    SimulationState,
)
import random

from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
    UTC_TIMEZONE,
)


class TestDiscreteEventSimulationEngine(unittest.TestCase):
    def setUp(self):
        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)
        self.svc_pt = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=10.0, std_service_minutes=2.0)
        self.svc_wc = ServiceConfigSnapshot(2, "WC", "Water Connection", mean_service_minutes=15.0, std_service_minutes=3.0)

    # =========================================================================
    # A. Empty system
    # =========================================================================
    def test_a_empty_system(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=60.0, seed=42)
        result = engine.run(state)

        self.assertEqual(result.total_customers_served, 0)
        self.assertEqual(result.total_waiting_person_minutes, 0.0)
        self.assertEqual(result.served_waiting_person_minutes, 0.0)
        self.assertEqual(result.unserved_waiting_person_minutes, 0.0)
        self.assertEqual(result.avg_waiting_time_minutes, 0.0)

    # =========================================================================
    # B. Existing waiting queue
    # =========================================================================
    def test_b_existing_waiting_queue_preserves_fifo(self):
        # Deterministic 5-minute service
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=5.0, std_service_minutes=0.0)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)

        # Token 1 arrived at -10m, Token 2 arrived at -5m
        w1 = QueueCitizenSnapshot(101, "PT-101", self.t_snap - timedelta(minutes=10))
        w2 = QueueCitizenSnapshot(102, "PT-102", self.t_snap - timedelta(minutes=5))

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (w1, w2), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=60.0, seed=42)
        result = engine.run(state)

        self.assertEqual(result.total_customers_served, 2)
        self.assertEqual(result.completed_citizen_ids, (101, 102))

        # Token 1 started at t=0 (wait = 0 - (-10) = 10m), finished at t=5
        # Token 2 started at t=5 (wait = 5 - (-5) = 10m), finished at t=10
        self.assertAlmostEqual(result.served_waiting_person_minutes, 20.0)
        self.assertAlmostEqual(result.avg_waiting_time_minutes, 10.0)

    # =========================================================================
    # C. Multiple services
    # =========================================================================
    def test_c_multiple_services_remain_independent(self):
        s1 = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=5.0, std_service_minutes=0.0)
        s2 = ServiceConfigSnapshot(2, "WC", "Water Connection", mean_service_minutes=8.0, std_service_minutes=0.0)

        # C1 assigned to PT, eligible ONLY for PT
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        # C2 assigned to WC, eligible ONLY for WC
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 2, frozenset([2]), None)

        pt_token = QueueCitizenSnapshot(11, "PT-11", self.t_snap)
        wc_token = QueueCitizenSnapshot(22, "WC-22", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: s1, 2: s2},
            queues={
                1: ServiceQueueSnapshot(1, (pt_token,), ()),
                2: ServiceQueueSnapshot(2, (wc_token,), ()),
            },
            counters={1: c1, 2: c2},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=30.0, seed=42)
        result = engine.run(state)

        self.assertEqual(result.total_customers_served, 2)
        self.assertEqual(result.service_metrics[1].customers_served, 1)
        self.assertEqual(result.service_metrics[2].customers_served, 1)

    # =========================================================================
    # D. Existing serving token
    # =========================================================================
    def test_d_existing_serving_token_respects_elapsed_time(self):
        # Service mean = 10m, std = 0.0m
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=10.0, std_service_minutes=0.0)

        # Serving token started 6 minutes ago (elapsed = 6.0m)
        serving_token = ServingTokenSnapshot(
            token_id=50,
            token_number="PT-050",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=self.t_snap - timedelta(minutes=8),
            serving_at_utc=self.t_snap - timedelta(minutes=6),
        )
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), serving_token)

        # Waiting token in queue
        w1 = QueueCitizenSnapshot(51, "PT-051", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (w1,), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=30.0, seed=42)
        result = engine.run(state)

        # Token 50 finishes at rem = 10 - 6 = 4.0m
        # Token 51 starts at t=4.0m and finishes at t=4.0 + 10.0 = 14.0m
        # Token 51 wait time = 4.0 - 0.0 = 4.0m
        self.assertEqual(result.total_customers_served, 2)
        self.assertEqual(result.completed_citizen_ids, (50, 51))
        self.assertAlmostEqual(result.service_metrics[1].waiting_person_minutes, 4.0)

    # =========================================================================
    # E. Service duration
    # =========================================================================
    def test_e_service_duration_reproducible_and_non_negative(self):
        rng1 = random.Random(42)
        proc1 = ServiceProcess(rng1)

        rng2 = random.Random(42)
        proc2 = ServiceProcess(rng2)

        # Sample 200 values from both
        samples1 = [proc1.sample_duration(mean=5.0, std=3.0) for _ in range(200)]
        samples2 = [proc2.sample_duration(mean=5.0, std=3.0) for _ in range(200)]

        self.assertEqual(samples1, samples2)
        for s in samples1:
            self.assertGreater(s, 0.0, "Service duration must never be non-positive")

        # Zero std returns exact mean
        self.assertEqual(proc1.sample_duration(mean=7.5, std=0.0), 7.5)

    # =========================================================================
    # F. Arrival process
    # =========================================================================
    def test_f_arrival_process_seed_behavior(self):
        rates = {1: 12.0}  # 12 per hour

        p1 = PoissonArrivalProcess(rates, rng=random.Random(42))
        p2 = PoissonArrivalProcess(rates, rng=random.Random(42))
        p3 = PoissonArrivalProcess(rates, rng=random.Random(999))

        arr1 = [c.arrival_time_min for c in p1.generate_arrivals(horizon_minutes=60.0)]
        arr2 = [c.arrival_time_min for c in p2.generate_arrivals(horizon_minutes=60.0)]
        arr3 = [c.arrival_time_min for c in p3.generate_arrivals(horizon_minutes=60.0)]

        self.assertEqual(arr1, arr2)
        self.assertNotEqual(arr1, arr3)

    # =========================================================================
    # G. Waiting person-minutes (Manually calculable scenario)
    # =========================================================================
    def test_g_waiting_person_minutes_exact_calculation(self):
        """
        Single counter with deterministic service duration = 10.0m.
        Deterministic customer arrivals:
        - Customer 1 arrives at t = 0.0m -> served 0..10m -> wait = 0.0m
        - Customer 2 arrives at t = 2.0m -> served 10..20m -> wait = 8.0m (10 - 2)
        - Customer 3 arrives at t = 5.0m -> served 20..30m -> wait = 15.0m (20 - 5)
        Horizon = 35.0m.
        Expected Total Waiting Person-Minutes = 0.0 + 8.0 + 15.0 = 23.0 person-minutes.
        """
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=10.0, std_service_minutes=0.0)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)

        arr_proc = DeterministicArrivalProcess([
            (0.0, 1, "T1"),
            (2.0, 1, "T2"),
            (5.0, 1, "T3"),
        ])

        engine = DiscreteEventEngine(
            horizon_minutes=35.0,
            seed=42,
            arrival_process=arr_proc,
        )
        result = engine.run(state)

        self.assertEqual(result.total_customers_served, 3)
        self.assertAlmostEqual(result.total_waiting_person_minutes, 23.0)
        self.assertAlmostEqual(result.served_waiting_person_minutes, 23.0)
        self.assertAlmostEqual(result.unserved_waiting_person_minutes, 0.0)
        self.assertAlmostEqual(result.avg_waiting_time_minutes, 23.0 / 3.0)
        self.assertAlmostEqual(result.max_waiting_time_minutes, 15.0)

    # =========================================================================
    # H. Counter unavailable
    # =========================================================================
    def test_h_counter_unavailable_cannot_start_service(self):
        c_unavail = CounterSnapshot(1, 101, "C1", "UNAVAILABLE", self.t_snap, 1, frozenset([1]), None)
        w1 = QueueCitizenSnapshot(1, "T1", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (w1,), ())},
            counters={1: c_unavail},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=30.0, seed=42)
        result = engine.run(state)

        # Unavailable counter must not serve anyone
        self.assertEqual(result.total_customers_served, 0)
        self.assertEqual(result.service_metrics[1].unserved_customers_count, 1)

    # =========================================================================
    # I. Counter recovery
    # =========================================================================
    def test_i_counter_recovery_resumes_service(self):
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=5.0, std_service_minutes=0.0)
        c_init_unavail = CounterSnapshot(1, 101, "C1", "UNAVAILABLE", self.t_snap, 1, frozenset([1]), None)
        w1 = QueueCitizenSnapshot(1, "T1", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (w1,), ())},
            counters={1: c_init_unavail},
        )
        state = SimulationState.from_snapshot(snapshot)

        # Schedule counter to recover at t = 10.0m
        recovery_event = CounterAvailabilityEvent(time_min=10.0, counter_id=1, is_available=True)

        engine = DiscreteEventEngine(
            horizon_minutes=30.0,
            seed=42,
            availability_events=[recovery_event],
        )
        result = engine.run(state)

        # Counter resumes at t=10.0, serves w1 (wait = 10.0m)
        self.assertEqual(result.total_customers_served, 1)
        self.assertAlmostEqual(result.served_waiting_person_minutes, 10.0)

    # =========================================================================
    # J. Horizon
    # =========================================================================
    def test_j_horizon_termination_blocks_future_events(self):
        # Service duration = 15.0m
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=15.0, std_service_minutes=0.0)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)

        # Customer arrives at t=0, Customer 2 arrives at t=10
        # Customer 1 served 0..15. Customer 2 starts at t=15, scheduled completion at t=30.
        arr_proc = DeterministicArrivalProcess([(0.0, 1, "T1"), (10.0, 1, "T2")])

        # Horizon is 20.0m (Customer 2 completion at t=30 is after horizon!)
        engine = DiscreteEventEngine(horizon_minutes=20.0, seed=42, arrival_process=arr_proc)
        result = engine.run(state)

        # Only Customer 1 completed before horizon
        self.assertEqual(result.total_customers_served, 1)
        self.assertEqual(result.completed_citizen_ids, (10000,))

    # =========================================================================
    # K. Deep clone
    # =========================================================================
    def test_k_deep_clone_isolation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w1 = QueueCitizenSnapshot(1, "T1", self.t_snap)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (w1,), ()),
                2: ServiceQueueSnapshot(2, (), ()),
            },
            counters={1: c1},
        )
        state1 = SimulationState.from_snapshot(snapshot)
        state2 = state1.clone()

        # Mutate state2
        state2.counters[1].assigned_service_id = 2
        state2.queues[1].pop_left()

        # State1 must remain completely unaffected
        self.assertEqual(state1.counters[1].assigned_service_id, 1)
        self.assertEqual(state1.queues[1].length(), 1)

    # =========================================================================
    # L. Determinism
    # =========================================================================
    def test_l_determinism_identical_results(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)

        # Run 1
        arr1 = PoissonArrivalProcess({1: 15.0}, rng=random.Random(42))
        eng1 = DiscreteEventEngine(horizon_minutes=60.0, seed=42, arrival_process=arr1)
        res1 = eng1.run(state)

        # Run 2
        arr2 = PoissonArrivalProcess({1: 15.0}, rng=random.Random(42))
        eng2 = DiscreteEventEngine(horizon_minutes=60.0, seed=42, arrival_process=arr2)
        res2 = eng2.run(state)

        self.assertEqual(res1.total_customers_served, res2.total_customers_served)
        self.assertAlmostEqual(res1.total_waiting_person_minutes, res2.total_waiting_person_minutes)
        self.assertEqual(res1.completed_citizen_ids, res2.completed_citizen_ids)
        self.assertEqual(res1.queue_length_history, res2.queue_length_history)

    # =========================================================================
    # M. Multi-counter behavior
    # =========================================================================
    def test_m_multi_counter_parallel_dispatch(self):
        svc_det = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=10.0, std_service_minutes=0.0)
        # 2 counters assigned to same service
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 1, frozenset([1]), None)

        w1 = QueueCitizenSnapshot(1, "T1", self.t_snap)
        w2 = QueueCitizenSnapshot(2, "T2", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: svc_det},
            queues={1: ServiceQueueSnapshot(1, (w1, w2), ())},
            counters={1: c1, 2: c2},
        )
        state = SimulationState.from_snapshot(snapshot)
        engine = DiscreteEventEngine(horizon_minutes=30.0, seed=42)
        result = engine.run(state)

        # Both dispatched at t=0 concurrently!
        # Both finish at t=10.0 with 0 wait time
        self.assertEqual(result.total_customers_served, 2)
        self.assertAlmostEqual(result.served_waiting_person_minutes, 0.0)

    # =========================================================================
    # N. Future intervention hook
    # =========================================================================
    def test_n_future_intervention_hook(self):
        s1 = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=5.0, std_service_minutes=0.0)
        s2 = ServiceConfigSnapshot(2, "WC", "Water Connection", mean_service_minutes=5.0, std_service_minutes=0.0)

        # C1 is eligible for BOTH PT (1) and WC (2), but initially assigned to PT (1)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)

        # PT has 1 customer, WC has 1 customer
        w_pt = QueueCitizenSnapshot(1, "PT-1", self.t_snap)
        w_wc = QueueCitizenSnapshot(2, "WC-1", self.t_snap)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: s1, 2: s2},
            queues={
                1: ServiceQueueSnapshot(1, (w_pt,), ()),
                2: ServiceQueueSnapshot(2, (w_wc,), ()),
            },
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)

        # At t=10.0, reassign C1 from PT (1) to WC (2)
        interv = CounterReassignIntervention(time_min=10.0, counter_id=1, new_service_id=2)

        engine = DiscreteEventEngine(horizon_minutes=30.0, seed=42, interventions=[interv])
        result = engine.run(state)

        # C1 serves PT customer first (0..5m).
        # C1 is idle from 5..10m.
        # At t=10m, C1 is reassigned to WC (2) and immediately starts serving WC customer (10..15m)!
        self.assertEqual(result.total_customers_served, 2)
        self.assertEqual(result.service_metrics[1].customers_served, 1)
        self.assertEqual(result.service_metrics[2].customers_served, 1)

    def test_n_invalid_intervention_raises_error(self):
        # C1 is only eligible for service 1
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={1: ServiceQueueSnapshot(1, (), ()), 2: ServiceQueueSnapshot(2, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snapshot)

        # Attempt to reassign C1 to service 2 (ineligible!)
        interv = CounterReassignIntervention(time_min=5.0, counter_id=1, new_service_id=2)
        engine = DiscreteEventEngine(horizon_minutes=20.0, seed=42, interventions=[interv])

        with self.assertRaises(ValueError):
            engine.run(state)


if __name__ == "__main__":
    unittest.main()
