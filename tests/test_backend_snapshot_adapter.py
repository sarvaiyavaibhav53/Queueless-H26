"""
PRAVAAH — Step 2C.2: Physical Counter Database Model & Backend Snapshot Adapter Tests.

Focused unit/integration tests verifying:
A. Counter model persistence.
B. CounterService eligibility persistence.
C. Duplicate counter/service eligibility rejection.
D. Counter assigned service representation.
E. Counter eligibility mapping to eligible_service_ids.
F. WAITING tokens mapping into waiting_tokens.
G. FIFO ordering preservation.
H. CALLED token mapping.
I. SERVING token mapping.
J. Idle counter producing current_token=None.
K. Counter status mapping.
L. Service configuration mapping.
M. Recent arrival timestamps from actual token creation times.
N. Explicit snapshot_time_utc respected.
O. No current-time call required inside adapter.
P. Invalid assigned-service/eligibility state fails loudly.
Q. Resulting snapshot passes directly into compute_raw_features().
R. Canonical Step 2C.1 fixture reproduces exact expected raw features through adapter.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

# Ensure project root and backend are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.models.base import Base
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.services.snapshot_adapter import (
    SnapshotAdapterError,
    build_simulation_snapshot,
)
from shared.features import (
    RAW_FEATURE_NAMES,
    UTC_TIMEZONE,
    compute_raw_features,
    raw_features_to_dict,
)


class TestPhysicalCounterAndSnapshotAdapter(unittest.TestCase):
    def setUp(self):
        # Create an isolated in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    # =========================================================================
    # A. Counter model persistence
    # =========================================================================
    def test_a_counter_model_persistence(self):
        office = Office(id=1, code="AMD", name="Ahmedabad Municipal", counters=2)
        service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        self.db.add_all([office, service])
        self.db.commit()

        status_time = self.t_snap - timedelta(hours=1)
        counter = Counter(
            id=10,
            office_id=1,
            name="Counter 10",
            status=CounterStatus.ACTIVE,
            assigned_service_id=1,
            status_changed_at=status_time,
        )
        self.db.add(counter)
        self.db.commit()

        queried = self.db.query(Counter).filter(Counter.id == 10).first()
        self.assertIsNotNone(queried)
        self.assertEqual(queried.office_id, 1)
        self.assertEqual(queried.name, "Counter 10")
        self.assertEqual(queried.status, CounterStatus.ACTIVE)
        self.assertEqual(queried.assigned_service_id, 1)
        status_time_utc = queried.status_changed_at if queried.status_changed_at.tzinfo else queried.status_changed_at.replace(tzinfo=timezone.utc)
        self.assertEqual(status_time_utc, status_time)

    # =========================================================================
    # B. CounterService eligibility persistence
    # =========================================================================
    def test_b_counter_service_eligibility_persistence(self):
        office = Office(id=1, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        s2 = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)
        c = Counter(
            id=1, office_id=1, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        self.db.add_all([office, s1, s2, c])
        self.db.commit()

        cs1 = CounterService(counter_id=1, service_id=1)
        cs2 = CounterService(counter_id=1, service_id=2)
        self.db.add_all([cs1, cs2])
        self.db.commit()

        eligibilities = self.db.query(CounterService).filter(CounterService.counter_id == 1).all()
        self.assertEqual(len(eligibilities), 2)
        service_ids = {e.service_id for e in eligibilities}
        self.assertEqual(service_ids, {1, 2})

    # =========================================================================
    # C. Duplicate counter/service eligibility is rejected
    # =========================================================================
    def test_c_duplicate_counter_service_rejected(self):
        office = Office(id=1, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c = Counter(
            id=1, office_id=1, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        self.db.add_all([office, s1, c])
        self.db.commit()

        cs_first = CounterService(counter_id=1, service_id=1)
        self.db.add(cs_first)
        self.db.commit()

        # Attempt duplicate insertion in a separate session to test DB constraint cleanly
        db2 = self.Session()
        try:
            cs_dup = CounterService(counter_id=1, service_id=1)
            db2.add(cs_dup)
            with self.assertRaises(IntegrityError):
                db2.commit()
        finally:
            db2.rollback()
            db2.close()

    # =========================================================================
    # D. Counter assigned service is represented correctly
    # =========================================================================
    def test_d_counter_assigned_service_representation(self):
        office = Office(id=1, code="AMD", name="Ahmedabad", counters=1)
        service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        counter = Counter(
            id=1, office_id=1, name="Desk 1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        self.db.add_all([office, service, counter])
        self.db.commit()

        fetched = self.db.query(Counter).filter(Counter.id == 1).first()
        self.assertEqual(fetched.assigned_service.id, 1)
        self.assertEqual(fetched.assigned_service.code, "PT")

    # =========================================================================
    # E. Counter eligibility maps correctly into eligible_service_ids
    # =========================================================================
    def test_e_counter_eligibility_maps_into_snapshot(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        s2 = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)
        c = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        self.db.add_all([office, s1, s2, c])
        self.db.commit()

        self.db.add_all([
            CounterService(counter_id=1, service_id=1),
            CounterService(counter_id=1, service_id=2),
        ])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        counter_snap = snapshot.counters[1]
        self.assertEqual(counter_snap.eligible_service_ids, frozenset([1, 2]))
        self.assertEqual(counter_snap.assigned_service_id, 1)

    # =========================================================================
    # F. WAITING tokens map into waiting_tokens
    # =========================================================================
    def test_f_waiting_tokens_map_into_waiting_tokens(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        t1 = QueueToken(
            id=101, token_number="AMD-PT-001", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=self.t_snap - timedelta(minutes=15)
        )
        t2 = QueueToken(
            id=102, token_number="AMD-PT-002", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=self.t_snap - timedelta(minutes=10)
        )
        self.db.add_all([office, s1, c1, cs1, t1, t2])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        queue_pt = snapshot.queues[1]
        self.assertEqual(len(queue_pt.waiting_tokens), 2)
        self.assertEqual(queue_pt.waiting_tokens[0].token_id, 101)
        self.assertEqual(queue_pt.waiting_tokens[0].token_number, "AMD-PT-001")
        self.assertEqual(queue_pt.waiting_tokens[1].token_id, 102)

    # =========================================================================
    # G. FIFO ordering is preserved
    # =========================================================================
    def test_g_fifo_ordering_is_strictly_preserved(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        # Insert out of chronological order into DB
        t_newer = QueueToken(
            id=200, token_number="T-NEW", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=self.t_snap - timedelta(minutes=2)
        )
        t_older = QueueToken(
            id=100, token_number="T-OLD", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=self.t_snap - timedelta(minutes=20)
        )
        self.db.add_all([office, s1, c1, cs1, t_newer, t_older])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        waiting = snapshot.queues[1].waiting_tokens
        self.assertEqual(len(waiting), 2)
        self.assertEqual(waiting[0].token_id, 100)
        self.assertEqual(waiting[1].token_id, 200)
        self.assertLess(waiting[0].arrival_time_utc, waiting[1].arrival_time_utc)

    # =========================================================================
    # H. CALLED token maps correctly
    # =========================================================================
    def test_h_called_token_maps_correctly(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        called_time = self.t_snap - timedelta(minutes=2)
        tok = QueueToken(
            id=50, token_number="PT-050", office_id=101, service_id=1,
            counter_id=1, status=TokenStatus.CALLED,
            created_at=self.t_snap - timedelta(minutes=10),
            called_at=called_time, serving_at=None
        )
        self.db.add_all([office, s1, c1, cs1, tok])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        counter_snap = snapshot.counters[1]
        self.assertIsNotNone(counter_snap.current_token)
        self.assertEqual(counter_snap.current_token.token_id, 50)
        self.assertEqual(counter_snap.current_token.status, "CALLED")
        self.assertEqual(counter_snap.current_token.called_at_utc, called_time)
        self.assertIsNone(counter_snap.current_token.serving_at_utc)

    # =========================================================================
    # I. SERVING token maps correctly
    # =========================================================================
    def test_i_serving_token_maps_correctly(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        called_time = self.t_snap - timedelta(minutes=8)
        serving_time = self.t_snap - timedelta(minutes=6)
        tok = QueueToken(
            id=51, token_number="PT-051", office_id=101, service_id=1,
            counter_id=1, status=TokenStatus.SERVING,
            created_at=self.t_snap - timedelta(minutes=20),
            called_at=called_time, serving_at=serving_time
        )
        self.db.add_all([office, s1, c1, cs1, tok])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        counter_snap = snapshot.counters[1]
        self.assertIsNotNone(counter_snap.current_token)
        self.assertEqual(counter_snap.current_token.token_id, 51)
        self.assertEqual(counter_snap.current_token.status, "SERVING")
        self.assertEqual(counter_snap.current_token.called_at_utc, called_time)
        self.assertEqual(counter_snap.current_token.serving_at_utc, serving_time)

    # =========================================================================
    # J. Idle counter produces current_token=None
    # =========================================================================
    def test_j_idle_counter_produces_none(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        self.db.add_all([office, s1, c1, cs1])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        counter_snap = snapshot.counters[1]
        self.assertIsNone(counter_snap.current_token)
        self.assertTrue(counter_snap.is_available)
        self.assertFalse(counter_snap.is_busy)

    # =========================================================================
    # K. Counter status maps correctly
    # =========================================================================
    def test_k_counter_status_maps_correctly(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=2)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c_active = Counter(
            id=1, office_id=101, name="Active Desk", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        c_unavail = Counter(
            id=2, office_id=101, name="Unavailable Desk", status=CounterStatus.UNAVAILABLE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        cs2 = CounterService(counter_id=2, service_id=1)
        self.db.add_all([office, s1, c_active, c_unavail, cs1, cs2])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        self.assertEqual(snapshot.counters[1].status, "ACTIVE")
        self.assertEqual(snapshot.counters[2].status, "UNAVAILABLE")

    # =========================================================================
    # L. Service configuration maps correctly
    # =========================================================================
    def test_l_service_configuration_maps_correctly(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=12.5, std_minutes=3.5)
        s2 = Service(id=2, code="WC", name="Water Connection", mean_minutes=8.0, std_minutes=None)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        self.db.add_all([office, s1, s2, c1, cs1])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        self.assertIn(1, snapshot.services)
        self.assertIn(2, snapshot.services)
        self.assertEqual(snapshot.services[1].mean_service_minutes, 12.5)
        self.assertEqual(snapshot.services[1].std_service_minutes, 3.5)
        self.assertEqual(snapshot.services[2].mean_service_minutes, 8.0)
        self.assertEqual(snapshot.services[2].std_service_minutes, 0.0)  # std_minutes=None defaults to 0.0

    # =========================================================================
    # M. Recent arrival timestamps come from actual token creation times
    # =========================================================================
    def test_m_recent_arrivals_from_actual_creation_times(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)

        # Tokens in window (t_snap - 60m, t_snap]
        arr_20m_ago = self.t_snap - timedelta(minutes=20)
        arr_50m_ago = self.t_snap - timedelta(minutes=50)
        # Token older than 60m
        arr_70m_ago = self.t_snap - timedelta(minutes=70)

        t1 = QueueToken(
            id=1, token_number="T1", office_id=101, service_id=1,
            status=TokenStatus.COMPLETED, created_at=arr_50m_ago
        )
        t2 = QueueToken(
            id=2, token_number="T2", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=arr_20m_ago
        )
        t_old = QueueToken(
            id=3, token_number="T3", office_id=101, service_id=1,
            status=TokenStatus.COMPLETED, created_at=arr_70m_ago
        )
        self.db.add_all([office, s1, c1, cs1, t1, t2, t_old])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        recent_arr = snapshot.queues[1].recent_arrivals_utc

        # Should only include t1 and t2 (within 60m lookback), strictly excluding t_old
        self.assertEqual(len(recent_arr), 2)
        self.assertEqual(recent_arr[0], arr_50m_ago)
        self.assertEqual(recent_arr[1], arr_20m_ago)

    # =========================================================================
    # N. Explicit snapshot_time_utc is respected
    # =========================================================================
    def test_n_explicit_snapshot_time_utc_respected(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)

        t_past = self.t_snap - timedelta(minutes=10)
        t_future = self.t_snap + timedelta(minutes=10)

        tok_past = QueueToken(
            id=1, token_number="T1", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=t_past
        )
        tok_future = QueueToken(
            id=2, token_number="T2", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=t_future
        )
        self.db.add_all([office, s1, c1, cs1, tok_past, tok_future])
        self.db.commit()

        # At t_snap, tok_future must NOT be included in waiting queue
        snap_now = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        self.assertEqual(len(snap_now.queues[1].waiting_tokens), 1)
        self.assertEqual(snap_now.queues[1].waiting_tokens[0].token_id, 1)

        # Later snapshot at t_future + 5m includes both
        snap_later = build_simulation_snapshot(
            self.db, office_id=101, snapshot_time_utc=t_future + timedelta(minutes=5)
        )
        self.assertEqual(len(snap_later.queues[1].waiting_tokens), 2)

    # =========================================================================
    # O. No current-time call is required inside the adapter
    # =========================================================================
    def test_o_no_current_time_call(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        self.db.add_all([office, s1, c1, cs1])
        self.db.commit()

        # Verify statically via AST and source inspection that snapshot_adapter contains NO current-time calls
        import ast
        import inspect
        import app.services.snapshot_adapter as adapter_mod

        source = inspect.getsource(adapter_mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr,
                        ("now", "utcnow"),
                        f"Found forbidden current-time call '{node.func.attr}()' in snapshot adapter"
                    )
                elif isinstance(node.func, ast.Name):
                    self.assertNotIn(
                        node.func.id,
                        ("now", "utcnow"),
                        f"Found forbidden current-time call '{node.func.id}()' in snapshot adapter"
                    )

        # Verify adapter executes cleanly using explicit snapshot_time_utc
        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.snapshot_time_utc, self.t_snap)

    # =========================================================================
    # P. Invalid assigned-service/eligibility state fails loudly
    # =========================================================================
    def test_p_invalid_eligibility_state_fails_loudly(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        s2 = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)
        # Counter is assigned service 2 (WC), but only eligible for service 1 (PT)
        c1 = Counter(
            id=1, office_id=101, name="Corrupt Desk", status=CounterStatus.ACTIVE,
            assigned_service_id=2, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        self.db.add_all([office, s1, s2, c1, cs1])
        self.db.commit()

        with self.assertRaises(SnapshotAdapterError) as ctx:
            build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        self.assertIn("NOT in its eligible services", str(ctx.exception))

    # =========================================================================
    # Q. Resulting snapshot can be passed directly into compute_raw_features()
    # =========================================================================
    def test_q_snapshot_passes_to_compute_raw_features(self):
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=1)
        s1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap
        )
        cs1 = CounterService(counter_id=1, service_id=1)
        t1 = QueueToken(
            id=10, token_number="AMD-PT-010", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=self.t_snap - timedelta(minutes=5)
        )
        self.db.add_all([office, s1, c1, cs1, t1])
        self.db.commit()

        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)
        feat = compute_raw_features(snapshot, service_id=1, position=1)

        self.assertEqual(feat.office_id, "AMD")
        self.assertEqual(feat.service_id, "PT")
        self.assertEqual(feat.people_ahead, 0)
        self.assertEqual(feat.queue_length, 1)
        self.assertEqual(feat.assigned_active_counters, 1)

    # =========================================================================
    # R. Canonical Step 2C.1 fixture produces exact expected raw features
    # =========================================================================
    def test_r_canonical_step2c1_fixture_reproduced_through_adapter(self):
        """
        Recreates the canonical multi-counter, multi-service scenario from Step 2C.1:
        Office: AMD (101)
        Service 1: PT (mean=10.0, std=2.0)
        Service 2: WC (mean=15.0, std=3.0)
        Counters:
        - C1: ACTIVE, assigned PT (eligible PT, WC), SERVING PT-001 (called 10:20, serving 10:24) -> elapsed=6.0m
        - C2: ACTIVE, assigned PT (eligible PT), idle -> current_token=None
        - C3: ACTIVE, assigned WC (eligible WC), SERVING WC-001 (called 10:25, serving 10:27) -> elapsed=3.0m
        Waiting PT tokens:
        - PT-002 (10:15)
        - PT-003 (10:22)
        - PT-004 (10:28)
        Waiting WC tokens:
        - WC-002 (10:18)
        - WC-003 (10:25)
        Recent Arrivals:
        - PT: 14 total in 60m (8 in last 30m, 6 in older 30m)
        - WC: 5 total in 60m (3 in last 30m, 2 in older 30m)
        """
        office = Office(id=101, code="AMD", name="Ahmedabad", counters=3)
        s_pt = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        s_wc = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)
        self.db.add_all([office, s_pt, s_wc])
        self.db.commit()

        # Counters
        c1 = Counter(
            id=1, office_id=101, name="C1", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap - timedelta(hours=2)
        )
        c2 = Counter(
            id=2, office_id=101, name="C2", status=CounterStatus.ACTIVE,
            assigned_service_id=1, status_changed_at=self.t_snap - timedelta(hours=2)
        )
        c3 = Counter(
            id=3, office_id=101, name="C3", status=CounterStatus.ACTIVE,
            assigned_service_id=2, status_changed_at=self.t_snap - timedelta(hours=2)
        )
        self.db.add_all([c1, c2, c3])
        self.db.commit()

        # Eligibilities
        self.db.add_all([
            CounterService(counter_id=1, service_id=1),
            CounterService(counter_id=1, service_id=2),
            CounterService(counter_id=2, service_id=1),
            CounterService(counter_id=3, service_id=2),
        ])
        self.db.commit()

        # Serving tokens
        tok_c1 = QueueToken(
            id=101, token_number="PT-001", office_id=101, service_id=1,
            counter_id=1, status=TokenStatus.SERVING,
            created_at=datetime(2026, 10, 7, 10, 10, 0, tzinfo=UTC_TIMEZONE),
            called_at=datetime(2026, 10, 7, 10, 20, 0, tzinfo=UTC_TIMEZONE),
            serving_at=datetime(2026, 10, 7, 10, 24, 0, tzinfo=UTC_TIMEZONE),
        )
        tok_c3 = QueueToken(
            id=201, token_number="WC-001", office_id=101, service_id=2,
            counter_id=3, status=TokenStatus.SERVING,
            created_at=datetime(2026, 10, 7, 10, 12, 0, tzinfo=UTC_TIMEZONE),
            called_at=datetime(2026, 10, 7, 10, 25, 0, tzinfo=UTC_TIMEZONE),
            serving_at=datetime(2026, 10, 7, 10, 27, 0, tzinfo=UTC_TIMEZONE),
        )
        self.db.add_all([tok_c1, tok_c3])
        self.db.commit()

        # Waiting PT tokens (3 tokens)
        pt_w1 = QueueToken(
            id=102, token_number="PT-002", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=datetime(2026, 10, 7, 10, 15, 0, tzinfo=UTC_TIMEZONE)
        )
        pt_w2 = QueueToken(
            id=103, token_number="PT-003", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=datetime(2026, 10, 7, 10, 22, 0, tzinfo=UTC_TIMEZONE)
        )
        pt_w3 = QueueToken(
            id=104, token_number="PT-004", office_id=101, service_id=1,
            status=TokenStatus.WAITING, created_at=datetime(2026, 10, 7, 10, 28, 0, tzinfo=UTC_TIMEZONE)
        )
        # Waiting WC tokens (2 tokens)
        wc_w1 = QueueToken(
            id=202, token_number="WC-002", office_id=101, service_id=2,
            status=TokenStatus.WAITING, created_at=datetime(2026, 10, 7, 10, 18, 0, tzinfo=UTC_TIMEZONE)
        )
        wc_w2 = QueueToken(
            id=203, token_number="WC-003", office_id=101, service_id=2,
            status=TokenStatus.WAITING, created_at=datetime(2026, 10, 7, 10, 25, 0, tzinfo=UTC_TIMEZONE)
        )
        self.db.add_all([pt_w1, pt_w2, pt_w3, wc_w1, wc_w2])
        self.db.commit()

        # Recent arrivals:
        # PT: 8 in [10:00..10:30], 6 in [09:30..10:00]
        # (Note: tok_c1 + pt_w1 + pt_w2 + pt_w3 already account for 4 arrivals at 10:10, 10:15, 10:22, 10:28)
        # We add 4 more in [10:00..10:30] and 6 in [09:30..10:00] as completed historical tokens
        pt_arr_30m_extra = [2, 5, 8, 20]
        for idx, m in enumerate(pt_arr_30m_extra, start=110):
            self.db.add(QueueToken(
                id=idx, token_number=f"PT-ARR-{idx}", office_id=101, service_id=1,
                status=TokenStatus.COMPLETED, created_at=datetime(2026, 10, 7, 10, m, 0, tzinfo=UTC_TIMEZONE)
            ))
        pt_arr_older = [32, 36, 40, 45, 50, 55]
        for idx, m in enumerate(pt_arr_older, start=120):
            self.db.add(QueueToken(
                id=idx, token_number=f"PT-ARR-{idx}", office_id=101, service_id=1,
                status=TokenStatus.COMPLETED, created_at=datetime(2026, 10, 7, 9, m, 0, tzinfo=UTC_TIMEZONE)
            ))

        # WC arrivals: 3 in [10:00..10:30] (tok_c3, wc_w1, wc_w2 already exist at 10:12, 10:18, 10:25 = 3 arrivals)
        # We add 2 in older 30m [09:30..10:00]
        wc_arr_older = [35, 48]
        for idx, m in enumerate(wc_arr_older, start=210):
            self.db.add(QueueToken(
                id=idx, token_number=f"WC-ARR-{idx}", office_id=101, service_id=2,
                status=TokenStatus.COMPLETED, created_at=datetime(2026, 10, 7, 9, m, 0, tzinfo=UTC_TIMEZONE)
            ))
        self.db.commit()

        # Build snapshot via adapter
        snapshot = build_simulation_snapshot(self.db, office_id=101, snapshot_time_utc=self.t_snap)

        # Compute raw features for PT (service_id=1), position=3
        vec = compute_raw_features(snapshot, service_id=1, position=3)

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
        self.assertAlmostEqual(vec.arrival_rate_30m, 16.0)  # 8 / 0.5
        self.assertAlmostEqual(vec.arrival_rate_60m, 14.0)  # 14 / 1.0
        self.assertAlmostEqual(vec.service_capacity_mu, 12.0)  # 2 * (60 / 10.0)
        self.assertAlmostEqual(vec.traffic_intensity, 14.0 / 12.0, places=6)

        dict_rep = raw_features_to_dict(vec)
        self.assertEqual(tuple(dict_rep.keys()), RAW_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
