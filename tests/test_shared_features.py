"""
Focused Unit Tests for PRAVAAH Shared Snapshot and Raw Feature Contract.
"""

import unittest
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import math

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.features import (
    UTC_TIMEZONE,
    LOCAL_TIMEZONE,
    RAW_FEATURE_NAMES,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    CounterSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
    RawFeatureVector,
    compute_raw_features,
    raw_features_to_dict,
)


class TestSnapshotDataclassesAndInvariants(unittest.TestCase):
    def setUp(self):
        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)
        self.service_pt = ServiceConfigSnapshot(
            service_id=1,
            service_code="PT",
            service_name="Property Tax",
            mean_service_minutes=10.0,
            std_service_minutes=2.0,
        )

    # A. Dataclass construction
    def test_a_dataclass_construction(self):
        citizen = QueueCitizenSnapshot(
            token_id=10,
            token_number="AMD-PT-010",
            arrival_time_utc=self.t_snap - timedelta(minutes=5),
        )
        self.assertEqual(citizen.token_id, 10)
        self.assertEqual(citizen.token_number, "AMD-PT-010")

    # B. UTC validation
    def test_b_utc_validation_rejects_naive(self):
        naive_dt = datetime(2026, 10, 7, 10, 30, 0)
        with self.assertRaises(ValueError):
            QueueCitizenSnapshot(token_id=1, token_number="T1", arrival_time_utc=naive_dt)

    def test_b_utc_validation_rejects_non_utc_aware(self):
        ist_dt = datetime(2026, 10, 7, 16, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        with self.assertRaises(ValueError):
            QueueCitizenSnapshot(token_id=1, token_number="T1", arrival_time_utc=ist_dt)

    # C. Snapshot local time conversion
    def test_c_snapshot_local_time_conversion(self):
        # 10:30 UTC = 16:00 IST (UTC + 5:30)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={},
        )
        local_dt = snapshot.snapshot_time_local
        self.assertEqual(local_dt.hour, 16)
        self.assertEqual(local_dt.minute, 0)
        self.assertEqual(local_dt.weekday(), 2)  # Wednesday

    # D. Counter is_busy
    def test_d_counter_is_busy(self):
        counter_idle = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=None,
        )
        self.assertFalse(counter_idle.is_busy)

        serving_token = ServingTokenSnapshot(
            token_id=100,
            token_number="T-100",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=self.t_snap - timedelta(minutes=5),
            serving_at_utc=self.t_snap - timedelta(minutes=4),
        )
        counter_busy = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=serving_token,
        )
        self.assertTrue(counter_busy.is_busy)

    # E. Counter is_available
    def test_e_counter_is_available(self):
        counter_avail = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=None,
        )
        self.assertTrue(counter_avail.is_available)

        counter_unavail = CounterSnapshot(
            counter_id=2,
            office_id=101,
            name="Counter 2",
            status="UNAVAILABLE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=None,
        )
        self.assertFalse(counter_unavail.is_available)

    # F. CALLED elapsed service = 0
    def test_f_called_elapsed_service_zero(self):
        called_token = ServingTokenSnapshot(
            token_id=101,
            token_number="T-101",
            service_id=1,
            counter_id=1,
            status="CALLED",
            called_at_utc=self.t_snap - timedelta(minutes=3),
            serving_at_utc=None,
        )
        counter = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=called_token,
        )
        self.assertEqual(counter.elapsed_service_minutes(self.t_snap), 0.0)

    # G. SERVING elapsed service calculation
    def test_g_serving_elapsed_service_calculation(self):
        serving_token = ServingTokenSnapshot(
            token_id=102,
            token_number="T-102",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=self.t_snap - timedelta(minutes=10),
            serving_at_utc=self.t_snap - timedelta(minutes=8, seconds=30),
        )
        counter = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=serving_token,
        )
        self.assertAlmostEqual(counter.elapsed_service_minutes(self.t_snap), 8.5)

    # H. negative elapsed service clamps to 0
    def test_h_negative_elapsed_service_clamps_to_zero(self):
        # serving_at in future relative to snapshot time
        future_serving = ServingTokenSnapshot(
            token_id=103,
            token_number="T-103",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=self.t_snap + timedelta(minutes=1),
            serving_at_utc=self.t_snap + timedelta(minutes=2),
        )
        counter = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=future_serving,
        )
        self.assertEqual(counter.elapsed_service_minutes(self.t_snap), 0.0)

    # I. FIFO validation
    def test_i_fifo_validation(self):
        c1 = QueueCitizenSnapshot(1, "T1", self.t_snap - timedelta(minutes=10))
        c2 = QueueCitizenSnapshot(2, "T2", self.t_snap - timedelta(minutes=5))
        # Valid order
        ServiceQueueSnapshot(service_id=1, waiting_tokens=(c1, c2))

        # Invalid out-of-order queue
        with self.assertRaises(ValueError):
            ServiceQueueSnapshot(service_id=1, waiting_tokens=(c2, c1))

    # J. Counter assignment/eligibility invariant
    def test_j_counter_assignment_eligibility_invariant(self):
        # Service 2 is not in eligible_service_ids {1}
        with self.assertRaises(ValueError):
            CounterSnapshot(
                counter_id=1,
                office_id=101,
                name="Counter 1",
                status="ACTIVE",
                status_changed_at_utc=self.t_snap,
                assigned_service_id=2,
                eligible_service_ids=frozenset([1]),
                current_token=None,
            )

    # Token exclusivity invariant
    def test_token_exclusivity_invariant(self):
        t1 = QueueCitizenSnapshot(1, "T1", self.t_snap - timedelta(minutes=5))
        st1 = ServingTokenSnapshot(
            token_id=1,  # Same token ID as waiting token!
            token_number="T1",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=self.t_snap - timedelta(minutes=6),
            serving_at_utc=self.t_snap - timedelta(minutes=5),
        )
        c1 = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=self.t_snap - timedelta(hours=1),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=st1,
        )
        with self.assertRaises(ValueError):
            SimulationSnapshot(
                office_id=101,
                office_code="AMD",
                snapshot_time_utc=self.t_snap,
                services={1: self.service_pt},
                queues={1: ServiceQueueSnapshot(service_id=1, waiting_tokens=(t1,))},
                counters={1: c1},
            )


class TestFeatureCalculations(unittest.TestCase):
    def setUp(self):
        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)
        self.service_pt = ServiceConfigSnapshot(1, "PT", "Property Tax", 10.0, 2.0)
        self.service_wc = ServiceConfigSnapshot(2, "WC", "Water Connection", 15.0, 3.0)

    # K. people_ahead calculation
    def test_k_people_ahead_calculation(self):
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={},
        )
        feat_pos1 = compute_raw_features(snapshot, service_id=1, position=1)
        self.assertEqual(feat_pos1.people_ahead, 0)

        feat_pos5 = compute_raw_features(snapshot, service_id=1, position=5)
        self.assertEqual(feat_pos5.people_ahead, 4)

    # L. queue_length calculation
    def test_l_queue_length_calculation(self):
        c1 = QueueCitizenSnapshot(1, "T1", self.t_snap - timedelta(minutes=10))
        c2 = QueueCitizenSnapshot(2, "T2", self.t_snap - timedelta(minutes=5))
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1, waiting_tokens=(c1, c2))},
            counters={},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        self.assertEqual(feat.queue_length, 2)

    # M & N. assigned_active_counters and busy_assigned_counters
    def test_m_n_assigned_and_busy_counters(self):
        c1 = CounterSnapshot(
            1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]),
            ServingTokenSnapshot(10, "T10", 1, 1, "SERVING", self.t_snap - timedelta(minutes=5), self.t_snap - timedelta(minutes=4))
        )
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        c3_unavail = CounterSnapshot(3, 101, "C3", "UNAVAILABLE", self.t_snap, 1, frozenset([1]), None)
        c4_other_service = CounterSnapshot(4, 101, "C4", "ACTIVE", self.t_snap, 2, frozenset([2]), None)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt, 2: self.service_wc},
            queues={1: ServiceQueueSnapshot(service_id=1), 2: ServiceQueueSnapshot(service_id=2)},
            counters={1: c1, 2: c2, 3: c3_unavail, 4: c4_other_service},
        )
        feat_pt = compute_raw_features(snapshot, service_id=1, position=1)
        self.assertEqual(feat_pt.assigned_active_counters, 2)  # C1 and C2
        self.assertEqual(feat_pt.busy_assigned_counters, 1)      # C1 busy, C2 idle

    # O. max_elapsed_service_minutes
    def test_o_max_elapsed_service_minutes(self):
        # C1 serving for 3 mins, C2 serving for 7 mins, C3 idle
        c1 = CounterSnapshot(
            1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]),
            ServingTokenSnapshot(11, "T11", 1, 1, "SERVING", self.t_snap - timedelta(minutes=4), self.t_snap - timedelta(minutes=3))
        )
        c2 = CounterSnapshot(
            2, 101, "C2", "ACTIVE", self.t_snap, 1, frozenset([1]),
            ServingTokenSnapshot(12, "T12", 1, 2, "SERVING", self.t_snap - timedelta(minutes=8), self.t_snap - timedelta(minutes=7))
        )
        c3 = CounterSnapshot(3, 101, "C3", "ACTIVE", self.t_snap, 1, frozenset([1]), None)

        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={1: c1, 2: c2, 3: c3},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        # Must pick max (7.0 min), NOT sum (10.0 min)!
        self.assertAlmostEqual(feat.max_elapsed_service_minutes, 7.0)

    # P & Q. arrival_rate_30m and arrival_rate_60m half-open interval
    def test_p_q_arrival_rates_boundary_inclusion(self):
        recent = (
            self.t_snap - timedelta(minutes=61),  # excluded (>60m ago)
            self.t_snap - timedelta(minutes=60),  # excluded (boundary t-60 is open: t-60 < t_arr)
            self.t_snap - timedelta(minutes=50),  # in 60m only
            self.t_snap - timedelta(minutes=31),  # in 60m only
            self.t_snap - timedelta(minutes=30),  # excluded from 30m (open boundary), in 60m
            self.t_snap - timedelta(minutes=25),  # in 30m and 60m
            self.t_snap - timedelta(minutes=10),  # in 30m and 60m
            self.t_snap,                          # included in 30m and 60m (closed boundary <= t)
        )
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1, recent_arrivals_utc=recent)},
            counters={},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        # In (t-30m, t]: minutes 25, 10, 0 -> 3 arrivals -> 3 / 0.5 = 6.0
        self.assertAlmostEqual(feat.arrival_rate_30m, 6.0)
        # In (t-60m, t]: minutes 50, 31, 30, 25, 10, 0 -> 6 arrivals -> 6 / 1.0 = 6.0
        self.assertAlmostEqual(feat.arrival_rate_60m, 6.0)

    # R. service_capacity_mu
    def test_r_service_capacity_mu(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},  # mean = 10.0 min
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={1: c1, 2: c2},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        # 2 counters * (60 / 10.0) = 12.0 citizens/hr
        self.assertAlmostEqual(feat.service_capacity_mu, 12.0)

    # S. rho with normal capacity
    def test_s_rho_normal_capacity(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        arrivals = tuple(self.t_snap - timedelta(minutes=m) for m in range(1, 13))  # 12 arrivals in 60m
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},  # mean = 10 min -> 1 counter = 6.0 mu
            queues={1: ServiceQueueSnapshot(service_id=1, recent_arrivals_utc=arrivals)},
            counters={1: c1},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        # lambda_60 = 12.0, mu = 6.0 -> rho = 2.0
        self.assertAlmostEqual(feat.traffic_intensity, 2.0)

    # T. rho with mu=0 and lambda=0
    def test_t_rho_zero_mu_zero_lambda(self):
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1, recent_arrivals_utc=())},
            counters={},  # 0 active counters -> mu = 0.0
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        self.assertEqual(feat.service_capacity_mu, 0.0)
        self.assertEqual(feat.arrival_rate_60m, 0.0)
        self.assertEqual(feat.traffic_intensity, 0.0)

    # U. rho with mu=0 and lambda>0
    def test_u_rho_zero_mu_positive_lambda_finite_sentinel(self):
        arrivals = (self.t_snap - timedelta(minutes=5),)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1, recent_arrivals_utc=arrivals)},
            counters={},  # mu = 0.0
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        self.assertEqual(feat.service_capacity_mu, 0.0)
        self.assertEqual(feat.arrival_rate_60m, 1.0)
        self.assertEqual(feat.traffic_intensity, 99.0)

    # V. no inf/NaN
    def test_v_no_inf_or_nan(self):
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={},
        )
        feat = compute_raw_features(snapshot, service_id=1, position=1)
        for k, v in feat.to_dict().items():
            if isinstance(v, float):
                self.assertTrue(math.isfinite(v), f"Feature {k} produced non-finite float {v}")

    # W. deterministic repeated calculation
    def test_w_deterministic_repeated_calculation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={1: c1},
        )
        res1 = compute_raw_features(snapshot, service_id=1, position=2)
        res2 = compute_raw_features(snapshot, service_id=1, position=2)
        self.assertEqual(res1, res2)

    # X. invalid position rejection
    def test_x_invalid_position_rejection(self):
        snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt},
            queues={1: ServiceQueueSnapshot(service_id=1)},
            counters={},
        )
        with self.assertRaises(ValueError):
            compute_raw_features(snapshot, service_id=1, position=0)
        with self.assertRaises(ValueError):
            compute_raw_features(snapshot, service_id=1, position=-1)


class TestCanonicalFixtureAndZeroCapacity(unittest.TestCase):
    def setUp(self):
        # 14. CANONICAL FIXTURE:
        # Snapshot Time: 2026-10-07 10:30:00 UTC (16:00 IST)
        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)

        self.service_pt = ServiceConfigSnapshot(
            service_id=1,
            service_code="PT",
            service_name="Property Tax",
            mean_service_minutes=10.0,
            std_service_minutes=2.0,
        )
        self.service_wc = ServiceConfigSnapshot(
            service_id=2,
            service_code="WC",
            service_name="Water Connection",
            mean_service_minutes=15.0,
            std_service_minutes=3.0,
        )

        # C1: PT, eligible {1, 2}, ACTIVE, serving PT started 10:24 UTC (elapsed = 6.0 min)
        st_c1 = ServingTokenSnapshot(
            token_id=101,
            token_number="PT-001",
            service_id=1,
            counter_id=1,
            status="SERVING",
            called_at_utc=datetime(2026, 10, 7, 10, 23, 0, tzinfo=UTC_TIMEZONE),
            serving_at_utc=datetime(2026, 10, 7, 10, 24, 0, tzinfo=UTC_TIMEZONE),
        )
        self.c1 = CounterSnapshot(
            counter_id=1,
            office_id=101,
            name="Counter 1",
            status="ACTIVE",
            status_changed_at_utc=datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC_TIMEZONE),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1, 2]),
            current_token=st_c1,
        )

        # C2: PT, eligible {1}, ACTIVE, idle
        self.c2 = CounterSnapshot(
            counter_id=2,
            office_id=101,
            name="Counter 2",
            status="ACTIVE",
            status_changed_at_utc=datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC_TIMEZONE),
            assigned_service_id=1,
            eligible_service_ids=frozenset([1]),
            current_token=None,
        )

        # C3: WC, eligible {2}, ACTIVE, serving WC started 10:20 UTC (elapsed = 10.0 min)
        st_c3 = ServingTokenSnapshot(
            token_id=201,
            token_number="WC-001",
            service_id=2,
            counter_id=3,
            status="SERVING",
            called_at_utc=datetime(2026, 10, 7, 10, 19, 0, tzinfo=UTC_TIMEZONE),
            serving_at_utc=datetime(2026, 10, 7, 10, 20, 0, tzinfo=UTC_TIMEZONE),
        )
        self.c3 = CounterSnapshot(
            counter_id=3,
            office_id=101,
            name="Counter 3",
            status="ACTIVE",
            status_changed_at_utc=datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC_TIMEZONE),
            assigned_service_id=2,
            eligible_service_ids=frozenset([2]),
            current_token=st_c3,
        )

        # PT waiting: PT-002 (10:15), PT-003 (10:22), PT-004 (10:28)
        self.pt_waiting = (
            QueueCitizenSnapshot(102, "PT-002", datetime(2026, 10, 7, 10, 15, 0, tzinfo=UTC_TIMEZONE)),
            QueueCitizenSnapshot(103, "PT-003", datetime(2026, 10, 7, 10, 22, 0, tzinfo=UTC_TIMEZONE)),
            QueueCitizenSnapshot(104, "PT-004", datetime(2026, 10, 7, 10, 28, 0, tzinfo=UTC_TIMEZONE)),
        )

        # WC waiting: WC-002 (10:18), WC-003 (10:25)
        self.wc_waiting = (
            QueueCitizenSnapshot(202, "WC-002", datetime(2026, 10, 7, 10, 18, 0, tzinfo=UTC_TIMEZONE)),
            QueueCitizenSnapshot(203, "WC-003", datetime(2026, 10, 7, 10, 25, 0, tzinfo=UTC_TIMEZONE)),
        )

        # PT arrivals: 8 in last 30m, 14 in last 60m (so 6 in [09:30..10:00] + 8 in [10:00..10:30])
        pt_arr_30m = [datetime(2026, 10, 7, 10, m, 0, tzinfo=UTC_TIMEZONE) for m in [2, 5, 8, 12, 15, 20, 24, 28]]
        pt_arr_older_30m = [datetime(2026, 10, 7, 9, m, 0, tzinfo=UTC_TIMEZONE) for m in [32, 36, 40, 45, 50, 55]]
        self.pt_recent_arrivals = tuple(sorted(pt_arr_older_30m + pt_arr_30m))
        self.assertEqual(len(self.pt_recent_arrivals), 14)

        # WC arrivals: 3 in last 30m, 5 in last 60m (2 older + 3 recent)
        wc_arr_30m = [datetime(2026, 10, 7, 10, m, 0, tzinfo=UTC_TIMEZONE) for m in [5, 15, 25]]
        wc_arr_older_30m = [datetime(2026, 10, 7, 9, m, 0, tzinfo=UTC_TIMEZONE) for m in [35, 48]]
        self.wc_recent_arrivals = tuple(sorted(wc_arr_older_30m + wc_arr_30m))
        self.assertEqual(len(self.wc_recent_arrivals), 5)

        self.snapshot = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt, 2: self.service_wc},
            queues={
                1: ServiceQueueSnapshot(1, self.pt_waiting, self.pt_recent_arrivals),
                2: ServiceQueueSnapshot(2, self.wc_waiting, self.wc_recent_arrivals),
            },
            counters={1: self.c1, 2: self.c2, 3: self.c3},
        )

    def test_canonical_fixture_pt_citizen4_position3(self):
        # Evaluating service_id=1 (PT), position=3
        vec = compute_raw_features(self.snapshot, service_id=1, position=3)

        self.assertEqual(vec.office_id, "AMD")
        self.assertEqual(vec.service_id, "PT")
        self.assertEqual(vec.hour, 16)
        self.assertEqual(vec.day_of_week, 2)

        self.assertEqual(vec.people_ahead, 2)
        self.assertEqual(vec.queue_length, 3)

        self.assertEqual(vec.assigned_active_counters, 2)
        self.assertEqual(vec.busy_assigned_counters, 1)

        self.assertAlmostEqual(vec.max_elapsed_service_minutes, 6.0)

        self.assertAlmostEqual(vec.service_mean_minutes, 10.0)
        self.assertAlmostEqual(vec.service_std_minutes, 2.0)

        self.assertAlmostEqual(vec.arrival_rate_30m, 16.0)  # 8 / 0.5 = 16.0
        self.assertAlmostEqual(vec.arrival_rate_60m, 14.0)  # 14 / 1.0 = 14.0

        self.assertAlmostEqual(vec.service_capacity_mu, 12.0)  # 2 * (60 / 10.0) = 12.0
        self.assertAlmostEqual(vec.traffic_intensity, 14.0 / 12.0, places=6)
        self.assertAlmostEqual(vec.traffic_intensity, 1.1666667, places=6)

        # Verify export to dictionary matches RAW_FEATURE_NAMES
        dict_rep = raw_features_to_dict(vec)
        self.assertEqual(tuple(dict_rep.keys()), RAW_FEATURE_NAMES)

    def test_zero_capacity_scenarios(self):
        # 15. ZERO-CAPACITY TEST:
        # Case A: PT has zero active assigned counters, but lambda_60 > 0
        c1_unavail = CounterSnapshot(
            1, 101, "C1", "UNAVAILABLE", self.t_snap, 1, frozenset([1, 2]), None
        )
        c2_unavail = CounterSnapshot(
            2, 101, "C2", "UNAVAILABLE", self.t_snap, 1, frozenset([1]), None
        )
        snap_zero_cap = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt, 2: self.service_wc},
            queues={
                1: ServiceQueueSnapshot(1, self.pt_waiting, self.pt_recent_arrivals),
                2: ServiceQueueSnapshot(2, self.wc_waiting, self.wc_recent_arrivals),
            },
            counters={1: c1_unavail, 2: c2_unavail, 3: self.c3},
        )
        vec_zero_cap = compute_raw_features(snap_zero_cap, service_id=1, position=1)
        self.assertEqual(vec_zero_cap.assigned_active_counters, 0)
        self.assertEqual(vec_zero_cap.service_capacity_mu, 0.0)
        self.assertAlmostEqual(vec_zero_cap.arrival_rate_60m, 14.0)
        self.assertEqual(vec_zero_cap.traffic_intensity, 99.0)

        # Case B: PT has zero active assigned counters AND lambda_60 == 0
        snap_zero_both = SimulationSnapshot(
            office_id=101,
            office_code="AMD",
            snapshot_time_utc=self.t_snap,
            services={1: self.service_pt, 2: self.service_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, (), ()),
            },
            counters={1: c1_unavail, 2: c2_unavail, 3: self.c3},
        )
        vec_zero_both = compute_raw_features(snap_zero_both, service_id=1, position=1)
        self.assertEqual(vec_zero_both.service_capacity_mu, 0.0)
        self.assertEqual(vec_zero_both.arrival_rate_60m, 0.0)
        self.assertEqual(vec_zero_both.traffic_intensity, 0.0)


if __name__ == "__main__":
    unittest.main()
