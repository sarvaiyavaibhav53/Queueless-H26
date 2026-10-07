"""
PRAVAAH — Step 2D.2 Simulation Scenario & Data Configuration Tests.

Validates the simulation domain configuration, lognormal right-skewed service times,
dynamic capacity calculations, piecewise-constant phase & burst arrival modeling,
and five benchmark scenario fixtures.

Required Validations (1 - 19):
1. All five services exist with correct means.
2. All counters exist with correct eligibility.
3. Lognormal parameter derivation.
4. Service duration is non-negative.
5. Same seed produces same generated arrivals/service times.
6. Different seeds produce different realizations.
7. Capacity is derived from mean service time.
8. Removing a counter reduces capacity.
9. Target rho produces approximately intended arrival rate over long run.
10. Arrival burst increases arrival intensity during configured phase.
11. Healthy scenario does not continuously accumulate an unbounded queue.
12. PT pressure scenario creates queue pressure.
13. Donor constraint scenario does not rely on hard-coded winner.
14. Counter failure scenario reduces capacity.
15. Scenario configurations are deterministic and isolated.
16. No scenario modifies another scenario's state.
17. No database dependency exists inside scenario-generation logic.
18. No global RNG mutation.
19. No negative/NaN/infinite simulation values.
"""

from datetime import datetime, timezone
import math
import random
import unittest

from app.interventions.evaluator import InterventionEvaluator
from app.interventions.models import EvaluatorConfig, PolicyType
from app.simulation.engine import (
    CounterAvailabilityEvent,
    CounterReassignIntervention,
    DiscreteEventEngine,
)
from app.simulation.processes import (
    ArrivalBurst,
    ArrivalPhase,
    LognormalServiceProcess,
    PhaseArrivalProcess,
    ServiceProcess,
    derive_lognormal_params,
)
from app.simulation.scenarios import (
    DEFAULT_CV,
    OFFICE_MUNICIPALITY_CODE,
    OFFICE_MUNICIPALITY_ID,
    OFFICE_RTO_CODE,
    OFFICE_RTO_ID,
    SCENARIO_REGISTRY,
    SERVICE_BC_CODE,
    SERVICE_BC_ID,
    SERVICE_BC_MEAN,
    SERVICE_DR_CODE,
    SERVICE_DR_ID,
    SERVICE_DR_MEAN,
    SERVICE_OT_CODE,
    SERVICE_OT_ID,
    SERVICE_OT_MEAN,
    SERVICE_PT_CODE,
    SERVICE_PT_ID,
    SERVICE_PT_MEAN,
    SERVICE_WC_CODE,
    SERVICE_WC_ID,
    SERVICE_WC_MEAN,
    ScenarioConfig,
    build_rto_scenario_healthy,
    build_scenario_a_healthy,
    build_scenario_b_pt_pressure,
    build_scenario_c_donor_constraint,
    build_scenario_d_temporary_intervention,
    build_scenario_e_counter_failure,
    compute_all_service_capacities_mu,
    compute_counter_capacity_mu,
    compute_service_capacity_mu,
    get_scenario,
    get_standard_municipality_counters,
    get_standard_municipality_services,
    get_standard_rto_counters,
    get_standard_rto_services,
)
from app.simulation.state import SimCounter, SimulationState


class TestSimulationScenarios(unittest.TestCase):

    # =========================================================================
    # 1. All five services exist with correct means
    # =========================================================================
    def test_01_all_five_services_exist_with_correct_means(self):
        muni_services = get_standard_municipality_services()
        rto_services = get_standard_rto_services()

        # Total 5 services
        all_services = {**muni_services, **rto_services}
        self.assertEqual(len(all_services), 5)

        # Municipality services
        self.assertIn(SERVICE_PT_ID, muni_services)
        self.assertEqual(muni_services[SERVICE_PT_ID].service_code, "PT")
        self.assertEqual(muni_services[SERVICE_PT_ID].mean_service_minutes, 10.0)

        self.assertIn(SERVICE_WC_ID, muni_services)
        self.assertEqual(muni_services[SERVICE_WC_ID].service_code, "WC")
        self.assertEqual(muni_services[SERVICE_WC_ID].mean_service_minutes, 15.0)

        self.assertIn(SERVICE_BC_ID, muni_services)
        self.assertEqual(muni_services[SERVICE_BC_ID].service_code, "BC")
        self.assertEqual(muni_services[SERVICE_BC_ID].mean_service_minutes, 7.0)

        # RTO services
        self.assertIn(SERVICE_OT_ID, rto_services)
        self.assertEqual(rto_services[SERVICE_OT_ID].service_code, "OT")
        self.assertEqual(rto_services[SERVICE_OT_ID].mean_service_minutes, 14.0)

        self.assertIn(SERVICE_DR_ID, rto_services)
        self.assertEqual(rto_services[SERVICE_DR_ID].service_code, "DR")
        self.assertEqual(rto_services[SERVICE_DR_ID].mean_service_minutes, 8.0)

        # Verify NO Income Certificate exists
        codes = [s.service_code.upper() for s in all_services.values()]
        names = [s.service_name.lower() for s in all_services.values()]
        self.assertNotIn("IC", codes)
        self.assertFalse(any("income" in name for name in names))

    # =========================================================================
    # 2. All counters exist with correct eligibility
    # =========================================================================
    def test_02_all_counters_exist_with_correct_eligibility(self):
        muni_counters = {c.name: c for c in get_standard_municipality_counters()}
        rto_counters = {c.name: c for c in get_standard_rto_counters()}

        # Municipality counters: C1 through C6
        self.assertEqual(len(muni_counters), 6)
        c1 = muni_counters["C1"]
        self.assertEqual(c1.assigned_service_id, SERVICE_PT_ID)
        self.assertEqual(c1.eligible_service_ids, frozenset({SERVICE_PT_ID, SERVICE_BC_ID}))

        c2 = muni_counters["C2"]
        self.assertEqual(c2.assigned_service_id, SERVICE_PT_ID)
        self.assertEqual(c2.eligible_service_ids, frozenset({SERVICE_PT_ID, SERVICE_WC_ID}))

        c3 = muni_counters["C3"]
        self.assertEqual(c3.assigned_service_id, SERVICE_WC_ID)
        self.assertEqual(c3.eligible_service_ids, frozenset({SERVICE_WC_ID, SERVICE_PT_ID}))

        c4 = muni_counters["C4"]
        self.assertEqual(c4.assigned_service_id, SERVICE_WC_ID)
        self.assertEqual(c4.eligible_service_ids, frozenset({SERVICE_WC_ID}))

        c5 = muni_counters["C5"]
        self.assertEqual(c5.assigned_service_id, SERVICE_BC_ID)
        self.assertEqual(c5.eligible_service_ids, frozenset({SERVICE_BC_ID, SERVICE_PT_ID}))

        c6 = muni_counters["C6"]
        self.assertEqual(c6.assigned_service_id, SERVICE_BC_ID)
        self.assertEqual(c6.eligible_service_ids, frozenset({SERVICE_BC_ID}))

        # RTO counters: R1 through R3
        self.assertEqual(len(rto_counters), 3)
        r1 = rto_counters["R1"]
        self.assertEqual(r1.assigned_service_id, SERVICE_OT_ID)
        self.assertEqual(r1.eligible_service_ids, frozenset({SERVICE_OT_ID}))

        r2 = rto_counters["R2"]
        self.assertEqual(r2.assigned_service_id, SERVICE_OT_ID)
        self.assertEqual(r2.eligible_service_ids, frozenset({SERVICE_OT_ID, SERVICE_DR_ID}))

        r3 = rto_counters["R3"]
        self.assertEqual(r3.assigned_service_id, SERVICE_DR_ID)
        self.assertEqual(r3.eligible_service_ids, frozenset({SERVICE_DR_ID}))

    # =========================================================================
    # 3. Lognormal parameter derivation
    # =========================================================================
    def test_03_lognormal_parameter_derivation(self):
        # Test derivation across various means with CV = 0.40
        for mean in [7.0, 8.0, 10.0, 14.0, 15.0]:
            cv = 0.40
            mu_log, sigma_log = derive_lognormal_params(mean=mean, cv=cv)

            # Mathematical checks:
            # E[X] = exp(mu_log + sigma_log^2 / 2) == mean
            reconstructed_mean = math.exp(mu_log + 0.5 * (sigma_log ** 2))
            self.assertAlmostEqual(reconstructed_mean, mean, places=9)

            # CV = sqrt(exp(sigma_log^2) - 1) == cv
            reconstructed_cv = math.sqrt(math.exp(sigma_log ** 2) - 1.0)
            self.assertAlmostEqual(reconstructed_cv, cv, places=9)

        # Zero CV -> deterministic
        mu_zero, sigma_zero = derive_lognormal_params(mean=10.0, cv=0.0)
        self.assertEqual(sigma_zero, 0.0)
        self.assertAlmostEqual(math.exp(mu_zero), 10.0, places=9)

        # Invalid inputs raise ValueError
        with self.assertRaises(ValueError):
            derive_lognormal_params(mean=0.0, cv=0.40)
        with self.assertRaises(ValueError):
            derive_lognormal_params(mean=-5.0, cv=0.40)
        with self.assertRaises(ValueError):
            derive_lognormal_params(mean=10.0, cv=-0.1)

    # =========================================================================
    # 4. Service duration is non-negative
    # =========================================================================
    def test_04_service_duration_is_non_negative(self):
        rng = random.Random(12345)
        proc = LognormalServiceProcess(rng, default_cv=0.40)

        # Sample 2000 service durations across all services
        for mean in [7.0, 8.0, 10.0, 14.0, 15.0]:
            for _ in range(400):
                dur = proc.sample_duration(mean=mean)
                self.assertGreater(dur, 0.0, f"Duration {dur} was not strictly positive")
                self.assertTrue(math.isfinite(dur))
                self.assertFalse(math.isnan(dur))

            # Sample remaining duration
            for elapsed in [0.0, 2.0, 5.0, 20.0, 50.0]:
                rem = proc.sample_remaining_duration(mean=mean, elapsed=elapsed)
                self.assertGreater(rem, 0.0)
                self.assertTrue(math.isfinite(rem))

    # =========================================================================
    # 5. Same seed produces same generated arrivals/service times
    # =========================================================================
    def test_05_same_seed_produces_same_results(self):
        # 1. Service process reproducibility
        proc_a = LognormalServiceProcess(random.Random(42), default_cv=0.40)
        proc_b = LognormalServiceProcess(random.Random(42), default_cv=0.40)

        seq_a = [proc_a.sample_duration(mean=10.0) for _ in range(100)]
        seq_b = [proc_b.sample_duration(mean=10.0) for _ in range(100)]
        self.assertEqual(seq_a, seq_b)

        # 2. Arrival process reproducibility
        scen = build_scenario_a_healthy()
        state = scen.build_simulation_state()

        arr_proc_a = scen.build_arrival_process(state, rng=random.Random(101))
        arr_proc_b = scen.build_arrival_process(state, rng=random.Random(101))

        arrivals_a = arr_proc_a.generate_arrivals(horizon_minutes=60.0)
        arrivals_b = arr_proc_b.generate_arrivals(horizon_minutes=60.0)

        self.assertEqual(len(arrivals_a), len(arrivals_b))
        for ca, cb in zip(arrivals_a, arrivals_b):
            self.assertEqual(ca.token_id, cb.token_id)
            self.assertEqual(ca.service_id, cb.service_id)
            self.assertAlmostEqual(ca.arrival_time_min, cb.arrival_time_min, places=9)

    # =========================================================================
    # 6. Different seeds can produce different realizations
    # =========================================================================
    def test_06_different_seeds_produce_different_realizations(self):
        proc_1 = LognormalServiceProcess(random.Random(42), default_cv=0.40)
        proc_2 = LognormalServiceProcess(random.Random(99), default_cv=0.40)

        seq_1 = [proc_1.sample_duration(mean=10.0) for _ in range(20)]
        seq_2 = [proc_2.sample_duration(mean=10.0) for _ in range(20)]
        self.assertNotEqual(seq_1, seq_2)

    # =========================================================================
    # 7. Capacity is derived from mean service time
    # =========================================================================
    def test_07_capacity_is_derived_from_mean_service_time(self):
        # For a single counter: mu = 60 / mean
        self.assertAlmostEqual(compute_counter_capacity_mu(10.0), 6.00, places=4)
        self.assertAlmostEqual(compute_counter_capacity_mu(15.0), 4.00, places=4)
        self.assertAlmostEqual(compute_counter_capacity_mu(7.0), 60.0 / 7.0, places=4)
        self.assertAlmostEqual(compute_counter_capacity_mu(14.0), 60.0 / 14.0, places=4)
        self.assertAlmostEqual(compute_counter_capacity_mu(8.0), 7.50, places=4)

        # In Municipality standard state:
        # 2 counters for PT -> mu = 2 * 6.00 = 12.00
        # 2 counters for WC -> mu = 2 * 4.00 = 8.00
        # 2 counters for BC -> mu = 2 * (60/7) = 17.142857
        scen = build_scenario_a_healthy()
        state = scen.build_simulation_state()
        caps = compute_all_service_capacities_mu(state)

        self.assertAlmostEqual(caps[SERVICE_PT_ID], 12.00, places=4)
        self.assertAlmostEqual(caps[SERVICE_WC_ID], 8.00, places=4)
        self.assertAlmostEqual(caps[SERVICE_BC_ID], 120.0 / 7.0, places=4)

    # =========================================================================
    # 8. Removing a counter reduces capacity
    # =========================================================================
    def test_08_removing_a_counter_reduces_capacity(self):
        scen = build_scenario_a_healthy()
        state = scen.build_simulation_state()

        # Before failure: 2 counters for PT -> capacity 12.0
        cap_pt_before = compute_service_capacity_mu(state, SERVICE_PT_ID)
        self.assertEqual(cap_pt_before, 12.0)

        # Counter 1 becomes UNAVAILABLE
        state.counters[1].is_active = False

        # After failure: 1 counter for PT -> capacity drops to 6.0
        cap_pt_after = compute_service_capacity_mu(state, SERVICE_PT_ID)
        self.assertEqual(cap_pt_after, 6.0)

        # Reassign Counter 3 (WC) to PT
        state.counters[3].assigned_service_id = SERVICE_PT_ID
        # PT now has C2 + C3 = 2 active counters -> capacity 12.0
        self.assertEqual(compute_service_capacity_mu(state, SERVICE_PT_ID), 12.0)
        # WC now has only C4 = 1 active counter -> capacity drops from 8.0 to 4.0
        self.assertEqual(compute_service_capacity_mu(state, SERVICE_WC_ID), 4.0)

    # =========================================================================
    # 9. Target rho produces approximately intended arrival rate over long run
    # =========================================================================
    def test_09_target_rho_produces_intended_arrival_rate_over_long_run(self):
        # Long horizon: 500 hours (30,000 minutes)
        # For PT with capacity mu = 12.0 and target rho = 0.70:
        # Expected lambda = 0.70 * 12.0 = 8.40 citizens/hour
        # Over 500 hours, expected arrivals = 4,200
        horizon = 30000.0  # minutes = 500 hours
        phases = [
            ArrivalPhase(
                phase_name="long_test",
                start_min=0.0,
                end_min=horizon,
                target_rho_by_service={SERVICE_PT_ID: 0.70},
            )
        ]
        capacities = {SERVICE_PT_ID: 12.0}
        proc = PhaseArrivalProcess(
            phases=phases,
            service_capacities_per_hour=capacities,
            rng=random.Random(2026),
        )

        arrivals = proc.generate_arrivals(horizon_minutes=horizon)
        pt_arrivals = [c for c in arrivals if c.service_id == SERVICE_PT_ID]

        expected_count = 0.70 * 12.0 * (horizon / 60.0)  # 4200.0
        actual_count = len(pt_arrivals)

        # Relative error should be well within 3% over 500 hours
        rel_error = abs(actual_count - expected_count) / expected_count
        self.assertLess(rel_error, 0.03, f"Rel error {rel_error} exceeded 3% (actual={actual_count}, expected={expected_count})")

    # =========================================================================
    # 10. Arrival burst actually increases arrival intensity during configured phase
    # =========================================================================
    def test_10_arrival_burst_increases_intensity(self):
        horizon = 120.0
        phases = [
            ArrivalPhase(
                phase_name="steady",
                start_min=0.0,
                end_min=120.0,
                target_rho_by_service={SERVICE_BC_ID: 0.60},  # mu = 17.14 -> lambda ~ 10.28/hr
            )
        ]
        bursts = [
            ArrivalBurst(
                burst_name="surge",
                service_id=SERVICE_BC_ID,
                start_min=40.0,
                end_min=80.0,
                multiplier=2.5,  # 2.5x increase
            )
        ]
        proc = PhaseArrivalProcess(
            phases=phases,
            service_capacities_per_hour={SERVICE_BC_ID: 60.0 / 7.0 * 2},
            bursts=bursts,
            rng=random.Random(42),
        )

        # Rate checks
        rate_before = proc.get_arrival_rate_per_hour(SERVICE_BC_ID, time_min=20.0)
        rate_during = proc.get_arrival_rate_per_hour(SERVICE_BC_ID, time_min=60.0)
        rate_after = proc.get_arrival_rate_per_hour(SERVICE_BC_ID, time_min=100.0)

        self.assertAlmostEqual(rate_during, rate_before * 2.5, places=4)
        self.assertAlmostEqual(rate_before, rate_after, places=4)

        # Realization check over multiple long seeds
        count_pre_total = 0
        count_burst_total = 0
        for seed in range(10):
            p = PhaseArrivalProcess(
                phases=phases,
                service_capacities_per_hour={SERVICE_BC_ID: 60.0 / 7.0 * 2},
                bursts=bursts,
                rng=random.Random(seed),
            )
            arrs = p.generate_arrivals(horizon_minutes=120.0)
            count_pre_total += sum(1 for c in arrs if 0.0 <= c.arrival_time_min < 40.0)
            count_burst_total += sum(1 for c in arrs if 40.0 <= c.arrival_time_min < 80.0)

        # Burst window (40 min) must have substantially more arrivals than non-burst window (40 min)
        self.assertGreater(count_burst_total, count_pre_total * 2.0)

    # =========================================================================
    # 11. Healthy scenario does not continuously accumulate an unbounded queue
    # =========================================================================
    def test_11_healthy_scenario_remains_stable(self):
        scen = build_scenario_a_healthy()
        state = scen.build_simulation_state()
        arr_proc = scen.build_arrival_process(state, rng=random.Random(scen.base_seed))
        svc_proc = scen.build_service_process(rng=random.Random(scen.base_seed))

        engine = DiscreteEventEngine(
            horizon_minutes=scen.horizon_minutes,
            seed=scen.base_seed,
            arrival_process=arr_proc,
            service_process=svc_proc,
        )
        result = engine.run(state)

        # Check health:
        # Total waiting person-minutes remains reasonable (< 150 across entire office over 1 hour)
        self.assertLess(result.total_waiting_person_minutes, 150.0)

        # All completed customers have reasonable wait times
        for s_id, wait_times in result.service_waiting_times.items():
            for w in wait_times:
                self.assertLess(w, 30.0, f"Customer wait {w} exceeded healthy bound")

        # Evaluate with InterventionEvaluator: Baseline must be chosen!
        eval_config = scen.build_evaluator_config()
        evaluator = InterventionEvaluator(eval_config)
        eval_res = evaluator.evaluate(state)
        self.assertEqual(
            eval_res.selected_policy,
            PolicyType.BASELINE,
            "Healthy scenario must conclude NO INTERVENTION (Baseline) is optimal",
        )

    # =========================================================================
    # 12. PT pressure scenario creates queue pressure
    # =========================================================================
    def test_12_pt_pressure_scenario_creates_pressure(self):
        # Compare Healthy vs PT Pressure
        healthy_scen = build_scenario_a_healthy()
        pressure_scen = build_scenario_b_pt_pressure()

        h_state = healthy_scen.build_simulation_state()
        p_state = pressure_scen.build_simulation_state()

        h_engine = DiscreteEventEngine(
            horizon_minutes=60.0,
            seed=42,
            arrival_process=healthy_scen.build_arrival_process(h_state, rng=random.Random(42)),
            service_process=healthy_scen.build_service_process(rng=random.Random(42)),
        )
        p_engine = DiscreteEventEngine(
            horizon_minutes=60.0,
            seed=42,
            arrival_process=pressure_scen.build_arrival_process(p_state, rng=random.Random(42)),
            service_process=pressure_scen.build_service_process(rng=random.Random(42)),
        )

        h_res = h_engine.run(h_state)
        p_res = p_engine.run(p_state)

        # PT pressure must produce substantially higher PT wait times than healthy
        pt_h_metric = h_res.service_metrics[SERVICE_PT_ID]
        pt_p_metric = p_res.service_metrics[SERVICE_PT_ID]

        self.assertGreater(pt_p_metric.total_waiting_person_minutes, pt_h_metric.total_waiting_person_minutes * 2.0)
        self.assertGreater(pt_p_metric.avg_waiting_time_minutes, pt_h_metric.avg_waiting_time_minutes)

    # =========================================================================
    # 13. Donor constraint scenario does not rely on hard-coded winner
    # =========================================================================
    def test_13_donor_constraint_scenario_objective_differentiation(self):
        scen = build_scenario_c_donor_constraint()
        state = scen.build_simulation_state()

        # Simulate C3 -> PT (stealing from near-capacity WC)
        # vs C5 -> PT (stealing from slack BC)
        arr_proc = scen.build_arrival_process(state, rng=random.Random(42))
        svc_proc = scen.build_service_process(rng=random.Random(42))

        # 1. C3 -> PT
        eng_c3 = DiscreteEventEngine(
            horizon_minutes=60.0,
            seed=42,
            arrival_process=arr_proc,
            service_process=svc_proc,
            interventions=[
                CounterReassignIntervention(time_min=0.0, counter_id=3, new_service_id=SERVICE_PT_ID)
            ],
        )
        res_c3 = eng_c3.run(state)

        # 2. C5 -> PT
        eng_c5 = DiscreteEventEngine(
            horizon_minutes=60.0,
            seed=42,
            arrival_process=arr_proc,
            service_process=svc_proc,
            interventions=[
                CounterReassignIntervention(time_min=0.0, counter_id=5, new_service_id=SERVICE_PT_ID)
            ],
        )
        res_c5 = eng_c5.run(state)

        # WC performance under C3 reassignment: WC was donor and was already near capacity
        wc_wait_under_c3 = res_c3.service_metrics[SERVICE_WC_ID].total_waiting_person_minutes
        wc_wait_under_c5 = res_c5.service_metrics[SERVICE_WC_ID].total_waiting_person_minutes

        # Reassigning C3 must cause WC wait times to surge compared to C5 reassignment
        self.assertGreater(wc_wait_under_c3, wc_wait_under_c5 * 1.5)

        # Reassigning C5 from slack BC results in substantially better overall office performance
        self.assertLess(res_c5.total_waiting_person_minutes, res_c3.total_waiting_person_minutes)

    # =========================================================================
    # 14. Counter failure scenario reduces capacity
    # =========================================================================
    def test_14_counter_failure_scenario_reduces_capacity(self):
        scen = build_scenario_e_counter_failure()
        self.assertEqual(len(scen.availability_events), 1)
        avail_event = scen.availability_events[0]
        self.assertEqual(avail_event.counter_id, 1)
        self.assertFalse(avail_event.is_available)
        self.assertEqual(avail_event.time_min, 10.0)

        state = scen.build_simulation_state()
        engine = DiscreteEventEngine(
            horizon_minutes=scen.horizon_minutes,
            seed=scen.base_seed,
            arrival_process=scen.build_arrival_process(state, rng=random.Random(scen.base_seed)),
            service_process=scen.build_service_process(rng=random.Random(scen.base_seed)),
            availability_events=scen.availability_events,
        )
        res = engine.run(state)

        # Counter failure should result in elevated PT waiting person-minutes
        pt_metric = res.service_metrics[SERVICE_PT_ID]
        self.assertGreater(pt_metric.total_waiting_person_minutes, 0.0)
        max_q = max(length for _, s_id, length in res.queue_length_history if s_id == SERVICE_PT_ID)
        self.assertGreater(max_q, 1)

    # =========================================================================
    # 15. Scenario configurations are deterministic and isolated
    # =========================================================================
    def test_15_scenario_configurations_deterministic_and_isolated(self):
        scen_1 = build_scenario_a_healthy()
        scen_2 = build_scenario_a_healthy()

        self.assertEqual(scen_1.scenario_id, scen_2.scenario_id)
        self.assertEqual(scen_1.horizon_minutes, scen_2.horizon_minutes)
        self.assertEqual(scen_1.initial_queue_counts, scen_2.initial_queue_counts)

        # Registry lookups
        for scen_id in SCENARIO_REGISTRY.keys():
            scen_a = get_scenario(scen_id)
            scen_b = get_scenario(scen_id)
            self.assertEqual(scen_a.scenario_id, scen_b.scenario_id)
            self.assertIsNot(scen_a, scen_b, "get_scenario must return independent copies")

    # =========================================================================
    # 16. No scenario modifies another scenario's state
    # =========================================================================
    def test_16_no_scenario_modifies_another_scenario_state(self):
        scen_a = build_scenario_a_healthy()
        state_a = scen_a.build_simulation_state()

        scen_b = build_scenario_b_pt_pressure()
        state_b = scen_b.build_simulation_state()

        # Mutate state A
        state_a.counters[1].is_active = False
        state_a.queues[SERVICE_PT_ID].waiting_citizens.clear()

        # State B must remain unchanged
        self.assertTrue(state_b.counters[1].is_active)
        self.assertEqual(state_b.queues[SERVICE_PT_ID].length(), 8)

    # =========================================================================
    # 17. No database dependency exists inside scenario-generation logic
    # =========================================================================
    def test_17_no_database_dependency(self):
        # All scenario builders execute in pure memory without DB session
        for scen_id, builder in SCENARIO_REGISTRY.items():
            scen = builder()
            state = scen.build_simulation_state()
            self.assertIsInstance(state, SimulationState)
            self.assertGreaterEqual(len(state.services), 2)
            self.assertGreaterEqual(len(state.counters), 3)

    # =========================================================================
    # 18. No global RNG mutation
    # =========================================================================
    def test_18_no_global_rng_mutation(self):
        rng_state_before = random.getstate()

        # Run multiple scenario builds and simulation runs
        for builder in [build_scenario_a_healthy, build_scenario_b_pt_pressure]:
            scen = builder()
            state = scen.build_simulation_state()
            engine = DiscreteEventEngine(
                horizon_minutes=30.0,
                seed=42,
                arrival_process=scen.build_arrival_process(state, rng=random.Random(42)),
                service_process=scen.build_service_process(rng=random.Random(42)),
            )
            engine.run(state)

        rng_state_after = random.getstate()
        self.assertEqual(
            rng_state_before,
            rng_state_after,
            "Global random state must never be mutated during scenario simulation",
        )

    # =========================================================================
    # 19. No negative/NaN/infinite simulation values
    # =========================================================================
    def test_19_no_negative_nan_infinite_values(self):
        for scen_id, builder in SCENARIO_REGISTRY.items():
            scen = builder()
            state = scen.build_simulation_state()
            engine = DiscreteEventEngine(
                horizon_minutes=scen.horizon_minutes,
                seed=scen.base_seed,
                arrival_process=scen.build_arrival_process(state, rng=random.Random(scen.base_seed)),
                service_process=scen.build_service_process(rng=random.Random(scen.base_seed)),
                availability_events=scen.availability_events,
            )
            res = engine.run(state)

            self.assertTrue(math.isfinite(res.total_waiting_person_minutes))
            self.assertFalse(math.isnan(res.total_waiting_person_minutes))
            self.assertGreaterEqual(res.total_waiting_person_minutes, 0.0)

            for s_id, sm in res.service_metrics.items():
                self.assertTrue(math.isfinite(sm.total_waiting_person_minutes))
                self.assertFalse(math.isnan(sm.total_waiting_person_minutes))
                self.assertGreaterEqual(sm.total_waiting_person_minutes, 0.0)
                self.assertGreaterEqual(sm.avg_waiting_time_minutes, 0.0)
                self.assertGreaterEqual(sm.max_waiting_time_minutes, 0.0)
                self.assertGreaterEqual(sm.customers_served, 0)
                self.assertGreaterEqual(sm.unserved_customers_count, 0)


if __name__ == "__main__":
    unittest.main()
