"""
PRAVAAH — Step 2C.5 API Intervention Evaluation Integration Tests.

Validates the read-only, admin-protected intervention evaluation endpoint:
POST /api/v1/interventions/evaluate

Required Coverage (A - J):
A. Admin can evaluate an authorized office.
B. Citizen rejected (403 Forbidden).
C. Employee rejected (403 Forbidden).
D. Invalid office rejected (404 Not Found / 422 Validation Error).
E. Endpoint passes explicit point-in-time UTC snapshot time.
F. Response contains baseline, naive, and optimized metrics.
G. No intervention is falsely reported as applied (applied=False, is_simulation_only=True).
H. Evaluator failure is returned safely without internal details or stack traces.
I. Office scoping is enforced.
J. Endpoint does not persist any intervention or mutate database state.
Plus: Unauthenticated request rejected (401 Unauthorized).
"""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.security import create_access_token, get_password_hash
from app.main import app
from app.models.base import Base
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.user import User, UserRole
from app.services.snapshot_adapter import (
    SnapshotAdapterError,
    build_simulation_snapshot,
)


class TestApiInterventionsEvaluate(unittest.TestCase):
    def setUp(self):
        # 1. Setup isolated in-memory SQLite database with single connection pool
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self.db = self.Session()

        # 2. Configure dependency override for get_db
        def override_get_db():
            db_session = self.Session()
            try:
                yield db_session
            finally:
                db_session.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        # 3. Seed users with distinct roles
        self.admin_user = User(
            id=1,
            email="admin@gov.in",
            password_hash=get_password_hash("adminpass"),
            role=UserRole.admin,
            is_active=True,
        )
        self.employee_user = User(
            id=2,
            email="emp@gov.in",
            password_hash=get_password_hash("emppass"),
            role=UserRole.employee,
            is_active=True,
        )
        self.citizen_user = User(
            id=3,
            email="citizen@gov.in",
            password_hash=get_password_hash("citpass"),
            role=UserRole.citizen,
            is_active=True,
        )
        self.db.add_all([self.admin_user, self.employee_user, self.citizen_user])

        # 4. Generate JWT tokens for each user
        self.admin_token = create_access_token({"sub": "1"})
        self.employee_token = create_access_token({"sub": "2"})
        self.citizen_token = create_access_token({"sub": "3"})

        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        self.employee_headers = {"Authorization": f"Bearer {self.employee_token}"}
        self.citizen_headers = {"Authorization": f"Bearer {self.citizen_token}"}

        # 5. Seed operational office, services, and counters
        self.office = Office(id=1, code="AMD", name="Ahmedabad Municipal Office", counters=2)
        self.service1 = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
        self.service2 = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)

        self.db.add_all([self.office, self.service1, self.service2])
        self.db.commit()

        # Physical counters
        # Counter 1 assigned to Service 1; eligible for Service 1 and Service 2
        # Counter 2 assigned to Service 1; eligible for Service 1 and Service 2
        now_utc = datetime.now(timezone.utc)
        self.counter1 = Counter(
            id=101,
            office_id=1,
            name="Counter 101",
            status=CounterStatus.ACTIVE,
            assigned_service_id=1,
            status_changed_at=now_utc - timedelta(hours=1),
        )
        self.counter2 = Counter(
            id=102,
            office_id=1,
            name="Counter 102",
            status=CounterStatus.ACTIVE,
            assigned_service_id=1,
            status_changed_at=now_utc - timedelta(hours=1),
        )
        self.db.add_all([self.counter1, self.counter2])
        self.db.commit()

        # Counter service eligibilities
        self.db.add_all([
            CounterService(counter_id=101, service_id=1),
            CounterService(counter_id=101, service_id=2),
            CounterService(counter_id=102, service_id=1),
            CounterService(counter_id=102, service_id=2),
        ])

        # Waiting tokens for Service 2 (overloaded bottleneck)
        t_base = now_utc - timedelta(minutes=20)
        for i in range(1, 6):
            self.db.add(
                QueueToken(
                    token_number=f"WC-{i:03d}",
                    office_id=1,
                    service_id=2,
                    user_id=3,
                    status=TokenStatus.WAITING,
                    created_at=t_base + timedelta(minutes=i),
                )
            )
        self.db.commit()

    def tearDown(self):
        app.dependency_overrides.clear()
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    # =========================================================================
    # A. Admin can evaluate an authorized office
    # =========================================================================
    def test_a_admin_can_evaluate_authorized_office(self):
        payload = {
            "office_id": 1,
            "horizon_minutes": 30.0,
            "replications": 2,
            "base_seed": 42,
            "max_candidates": 5,
        }
        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json=payload,
            headers=self.admin_headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["office_id"], 1)
        self.assertIn("selected_policy", data)
        self.assertIn(data["selected_policy"], ["BASELINE", "NAIVE", "OPTIMIZED"])
        self.assertIn("total_waiting_person_minutes", data)
        self.assertIn("baseline_result", data)
        self.assertIn("naive_result", data)
        self.assertEqual(data["replication_count"], 2)
        self.assertEqual(data["horizon_minutes"], 30.0)
        self.assertEqual(data["base_seed"], 42)
        self.assertIn("advisory preview only", data["status_message"])

    # =========================================================================
    # B. Citizen rejected (403 Forbidden)
    # =========================================================================
    def test_b_citizen_rejected(self):
        payload = {"office_id": 1}
        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json=payload,
            headers=self.citizen_headers,
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Not enough permissions")

    # =========================================================================
    # C. Employee rejected (403 Forbidden)
    # =========================================================================
    def test_c_employee_rejected(self):
        payload = {"office_id": 1}
        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json=payload,
            headers=self.employee_headers,
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Not enough permissions")

    # =========================================================================
    # D. Invalid office rejected (404 Not Found / 422 Validation Error)
    # =========================================================================
    def test_d_invalid_office_rejected(self):
        # Non-existent office -> 404
        response_404 = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 99999},
            headers=self.admin_headers,
        )
        self.assertEqual(response_404.status_code, 404)
        self.assertIn("not found", response_404.json()["detail"].lower())

        # Negative office_id -> 422 Pydantic validation error
        response_422_neg = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": -1},
            headers=self.admin_headers,
        )
        self.assertEqual(response_422_neg.status_code, 422)

        # Zero office_id -> 422
        response_422_zero = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 0},
            headers=self.admin_headers,
        )
        self.assertEqual(response_422_zero.status_code, 422)

    # =========================================================================
    # E. Endpoint passes explicit point-in-time snapshot time
    # =========================================================================
    def test_e_endpoint_passes_explicit_snapshot_time(self):
        with patch("app.api.interventions.build_simulation_snapshot", wraps=build_simulation_snapshot) as spy:
            t_before = datetime.now(timezone.utc)
            response = self.client.post(
                "/api/v1/interventions/evaluate",
                json={"office_id": 1, "replications": 1, "horizon_minutes": 10.0},
                headers=self.admin_headers,
            )
            t_after = datetime.now(timezone.utc)

            self.assertEqual(response.status_code, 200)
            self.assertTrue(spy.called)

            # Inspect the snapshot_time_utc argument passed to build_simulation_snapshot
            _, kwargs = spy.call_args
            snapshot_time = kwargs.get("snapshot_time_utc")
            self.assertIsNotNone(snapshot_time)
            self.assertIsInstance(snapshot_time, datetime)
            self.assertIsNotNone(snapshot_time.tzinfo)
            self.assertEqual(snapshot_time.utcoffset(), timedelta(0))
            self.assertGreaterEqual(snapshot_time, t_before - timedelta(seconds=1))
            self.assertLessEqual(snapshot_time, t_after + timedelta(seconds=1))

    # =========================================================================
    # F. Response contains baseline, naive, and optimized results
    # =========================================================================
    def test_f_response_contains_baseline_naive_optimized_results(self):
        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json={
                "office_id": 1,
                "horizon_minutes": 45.0,
                "replications": 3,
                "base_seed": 777,
                "p90_thresholds": {1: 30.0, 2: 45.0},
            },
            headers=self.admin_headers,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Baseline evaluation check
        baseline = data["baseline_result"]
        self.assertEqual(baseline["policy_type"], "BASELINE")
        self.assertIsNone(baseline["candidate"])
        self.assertIn("mean_total_waiting_person_minutes", baseline)
        self.assertIn("p90_wait_by_service", baseline)
        self.assertEqual(baseline["replication_count"], 3)

        # Naive evaluation check
        naive = data["naive_result"]
        self.assertEqual(naive["policy_type"], "NAIVE")
        self.assertIn("mean_total_waiting_person_minutes", naive)
        self.assertIn("p90_wait_by_service", naive)
        self.assertEqual(naive["replication_count"], 3)

        # Comparison metrics
        self.assertIn("total_waiting_person_minutes", data)
        self.assertIn("improvement_vs_baseline", data)
        self.assertIn("pct_improvement_vs_baseline", data)
        self.assertIn("improvement_vs_naive", data)
        self.assertIn("pct_improvement_vs_naive", data)
        self.assertIn("per_service_p90", data)
        self.assertIn("configured_p90_thresholds", data)
        self.assertIn("explanation", data)
        self.assertIn("is_feasible", data)

        # Candidate evaluations list
        self.assertIsInstance(data["candidate_evaluations"], list)
        self.assertGreaterEqual(len(data["candidate_evaluations"]), 1)

    # =========================================================================
    # G. No intervention is falsely reported as applied
    # =========================================================================
    def test_g_no_intervention_is_falsely_reported_as_applied(self):
        c101_before = self.db.query(Counter).filter(Counter.id == 101).first()
        c102_before = self.db.query(Counter).filter(Counter.id == 102).first()
        c101_svc_before = c101_before.assigned_service_id
        c102_svc_before = c102_before.assigned_service_id

        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 1, "replications": 2},
            headers=self.admin_headers,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Explicit safety markers
        self.assertFalse(data["applied"], "Must never report applied=True")
        self.assertTrue(data["is_simulation_only"], "Must declare simulation-only mode")

        # Database state check
        c101_after = self.db.query(Counter).filter(Counter.id == 101).first()
        c102_after = self.db.query(Counter).filter(Counter.id == 102).first()
        self.assertEqual(c101_after.assigned_service_id, c101_svc_before)
        self.assertEqual(c102_after.assigned_service_id, c102_svc_before)

    # =========================================================================
    # H. Evaluator failure is returned safely without internal details
    # =========================================================================
    def test_h_evaluator_failure_returned_safely_without_internal_details(self):
        # 1. SnapshotAdapterError mapped to safe 422
        with patch("app.api.interventions.build_simulation_snapshot", side_effect=SnapshotAdapterError("No counters configured")):
            resp = self.client.post(
                "/api/v1/interventions/evaluate",
                json={"office_id": 1},
                headers=self.admin_headers,
            )
            self.assertEqual(resp.status_code, 422)
            self.assertIn("Simulation snapshot generation failed", resp.json()["detail"])
            # Ensure no python traceback string
            self.assertNotIn("Traceback", resp.text)

        # 2. Unexpected exception with internal DB password / path mapped to safe 500
        sensitive_error = Exception("FATAL: connection failed postgresql://secret_admin:my_secret_pass@192.168.1.55:5432/db")
        with patch("app.api.interventions.build_simulation_snapshot", side_effect=sensitive_error):
            resp = self.client.post(
                "/api/v1/interventions/evaluate",
                json={"office_id": 1},
                headers=self.admin_headers,
            )
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(
                resp.json()["detail"],
                "An unexpected internal error occurred while generating the simulation snapshot.",
            )
            # Ensure internal secrets and IP addresses are completely shielded
            self.assertNotIn("secret_admin", resp.text)
            self.assertNotIn("my_secret_pass", resp.text)
            self.assertNotIn("192.168.1.55", resp.text)
            self.assertNotIn("Traceback", resp.text)

    # =========================================================================
    # I. Office scoping is enforced
    # =========================================================================
    def test_i_office_scoping_enforced(self):
        # Second office with its own separate counter and no waiting tokens
        office2 = Office(id=2, code="SRT", name="Surat Office", counters=1)
        self.db.add(office2)
        self.db.commit()

        c201 = Counter(
            id=201,
            office_id=2,
            name="Counter 201",
            status=CounterStatus.ACTIVE,
            assigned_service_id=1,
            status_changed_at=datetime.now(timezone.utc),
        )
        self.db.add(c201)
        self.db.add(CounterService(counter_id=201, service_id=1))
        self.db.commit()

        # Evaluating office 2 strictly reflects office 2's data
        response_o2 = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 2, "replications": 1},
            headers=self.admin_headers,
        )
        self.assertEqual(response_o2.status_code, 200)
        data_o2 = response_o2.json()
        self.assertEqual(data_o2["office_id"], 2)
        # Office 2 has 0 waiting customers, so waiting person-minutes is 0
        self.assertEqual(data_o2["baseline_result"]["mean_total_waiting_person_minutes"], 0.0)

        # In contrast, evaluating office 1 reflects office 1's queues (> 0 wait)
        response_o1 = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 1, "replications": 1},
            headers=self.admin_headers,
        )
        self.assertEqual(response_o1.status_code, 200)
        data_o1 = response_o1.json()
        self.assertEqual(data_o1["office_id"], 1)
        self.assertGreater(data_o1["baseline_result"]["mean_total_waiting_person_minutes"], 0.0)

        # Non-existent office strictly rejected with 404
        response_nonexistent = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 8888},
            headers=self.admin_headers,
        )
        self.assertEqual(response_nonexistent.status_code, 404)
        self.assertIn("not found", response_nonexistent.json()["detail"].lower())

    # =========================================================================
    # J. Endpoint does not persist an intervention
    # =========================================================================
    def test_j_endpoint_does_not_persist_intervention(self):
        # Record baseline row counts and state across all entities
        offices_count_before = self.db.query(Office).count()
        services_count_before = self.db.query(Service).count()
        counters_count_before = self.db.query(Counter).count()
        counter_services_count_before = self.db.query(CounterService).count()
        tokens_count_before = self.db.query(QueueToken).count()
        users_count_before = self.db.query(User).count()

        response = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 1, "replications": 2},
            headers=self.admin_headers,
        )
        self.assertEqual(response.status_code, 200)

        # Confirm zero row creations or deletions
        self.assertEqual(self.db.query(Office).count(), offices_count_before)
        self.assertEqual(self.db.query(Service).count(), services_count_before)
        self.assertEqual(self.db.query(Counter).count(), counters_count_before)
        self.assertEqual(self.db.query(CounterService).count(), counter_services_count_before)
        self.assertEqual(self.db.query(QueueToken).count(), tokens_count_before)
        self.assertEqual(self.db.query(User).count(), users_count_before)

    # =========================================================================
    # Additional Security: Unauthenticated request rejected
    # =========================================================================
    def test_k_unauthenticated_request_rejected(self):
        # Missing auth header -> 401
        response_no_auth = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 1},
        )
        self.assertEqual(response_no_auth.status_code, 401)

        # Invalid auth header -> 401
        response_bad_token = self.client.post(
            "/api/v1/interventions/evaluate",
            json={"office_id": 1},
            headers={"Authorization": "Bearer invalid_garbage_token"},
        )
        self.assertEqual(response_bad_token.status_code, 401)


if __name__ == "__main__":
    unittest.main()
