"""
PRAVAAH — Step 2E.1A: ML Dataset Generation Proof.

Demonstrates the authoritative pipeline:
    Authoritative simulation
            ↓
    simulation state
            ↓
    SimulationSnapshot
            ↓
    compute_raw_features()
            ↓
    15-feature ML row
            ↓
    actual simulated future wait

Architectural Guarantees & Constraints:
- Pure discrete-event simulation via DiscreteEventEngine (NOT legacy event_engine.py).
- Feature contract strictly restricted to the 15 RAW_FEATURE_NAMES from shared/features.py.
- Target variable (target_wait_minutes) represents the actual simulated future waiting
  time from the snapshot moment until the target citizen begins service (service_start_min).
- Strict information boundary: features are computed only from snapshot state at t_snap.
  Zero future information is inspected during feature computation.
- Deterministic seed reproduction.
- Repository-relative output under ml/data/proof/ (does not overwrite legacy synthetic data).
- Frozen components preserved: zero changes to simulator, scenarios, evaluator, or shared features.
"""

from datetime import datetime, timezone, timedelta
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure repository root and backend directory are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")

for p in (REPO_ROOT, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.simulation.engine import DiscreteEventEngine
from app.simulation.scenarios import (
    OFFICE_MUNICIPALITY_ID,
    OFFICE_MUNICIPALITY_CODE,
    OFFICE_RTO_ID,
    OFFICE_RTO_CODE,
    SERVICE_PT_ID,
    SERVICE_WC_ID,
    SERVICE_BC_ID,
    SERVICE_OT_ID,
    SERVICE_DR_ID,
    get_standard_municipality_services,
    get_standard_municipality_counters,
    get_standard_rto_services,
    get_standard_rto_counters,
)
from app.simulation.state import SimulationState
from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
    compute_raw_features,
    RAW_FEATURE_NAMES,
)

DEFAULT_PROOF_OUTPUT_DIR = os.path.join(REPO_ROOT, "ml", "data", "proof")
DEFAULT_BASE_SEED = 42


def build_proof_snapshots(
    base_snapshot_time_utc: Optional[datetime] = None,
) -> List[Tuple[str, SimulationSnapshot]]:
    """
    Constructs deterministic, realistic SimulationSnapshot instances representing
    diverse operational regimes across Municipality and RTO offices:

    1. Municipality Healthy (PT: 1, WC: 1, BC: 1) -> 3 waiting tokens
    2. Municipality PT Backlog (PT: 8, WC: 1, BC: 1) -> 10 waiting tokens
    3. Municipality Active In-Progress (Busy C1 & C2 serving PT, PT queue: 5, WC: 1) -> 6 waiting tokens
    4. Municipality Donor Constraint (PT: 8, WC: 4, BC: 1) -> 13 waiting tokens
    5. Municipality Temporary Look-Ahead (PT: 6, WC: 1, BC: 1) -> 8 waiting tokens
    6. Municipality Counter Failure (PT: 2, WC: 1, BC: 1, C1 UNAVAILABLE) -> 4 waiting tokens
    7. RTO Healthy Operation (OT: 3, DR: 2) -> 5 waiting tokens

    Total waiting tokens: 3 + 10 + 6 + 13 + 8 + 4 + 5 = 49 rows.
    """
    t_snap = base_snapshot_time_utc or datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)
    muni_services = get_standard_municipality_services()
    rto_services = get_standard_rto_services()
    muni_counters = get_standard_municipality_counters()
    rto_counters = get_standard_rto_counters()

    snapshots: List[Tuple[str, SimulationSnapshot]] = []

    def make_muni_counters(
        active_ids: Optional[set] = None,
        serving_map: Optional[Dict[int, ServingTokenSnapshot]] = None,
    ) -> Dict[int, CounterSnapshot]:
        c_dict = {}
        for c in muni_counters:
            is_active = (c.counter_id in active_ids) if active_ids is not None else True
            curr_tok = serving_map.get(c.counter_id) if serving_map else None
            c_dict[c.counter_id] = CounterSnapshot(
                counter_id=c.counter_id,
                office_id=OFFICE_MUNICIPALITY_ID,
                name=c.name,
                status="ACTIVE" if is_active else "UNAVAILABLE",
                status_changed_at_utc=t_snap,
                assigned_service_id=c.assigned_service_id,
                eligible_service_ids=c.eligible_service_ids,
                current_token=curr_tok,
            )
        return c_dict

    def make_queues(
        services_dict: dict,
        queue_counts: Dict[int, int],
        recent_arrival_rates: Optional[Dict[int, float]] = None,
    ) -> Dict[int, ServiceQueueSnapshot]:
        q_dict = {}
        tok_id = 100
        for s_id, s_cfg in services_dict.items():
            count = queue_counts.get(s_id, 0)
            waiting_tokens = []
            recent_arrivals = []
            # Generate waiting tokens arriving before or up to snapshot instant
            for i in range(count):
                tok_id += 1
                arr_offset = -float((count - i) * 2.0)
                arr_utc = t_snap + timedelta(minutes=arr_offset)
                waiting_tokens.append(
                    QueueCitizenSnapshot(
                        token_id=tok_id,
                        token_number=f"{s_cfg.service_code}-{tok_id}",
                        arrival_time_utc=arr_utc,
                    )
                )
                recent_arrivals.append(arr_utc)

            # Add background recent arrivals in the trailing 60m to reflect traffic intensity
            if recent_arrival_rates and s_id in recent_arrival_rates:
                target_count_60 = int(round(recent_arrival_rates[s_id]))
                extra_needed = max(0, target_count_60 - len(recent_arrivals))
                for j in range(extra_needed):
                    arr_min_back = -float(10.0 + j * 3.0)
                    if arr_min_back >= -60.0:
                        recent_arrivals.append(t_snap + timedelta(minutes=arr_min_back))

            recent_arrivals.sort()
            q_dict[s_id] = ServiceQueueSnapshot(
                service_id=s_id,
                waiting_tokens=tuple(waiting_tokens),
                recent_arrivals_utc=tuple(recent_arrivals),
            )
        return q_dict

    # 1. Municipality Healthy (3 tokens)
    s1_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 1, SERVICE_WC_ID: 1, SERVICE_BC_ID: 1},
        {SERVICE_PT_ID: 8.0, SERVICE_WC_ID: 5.0, SERVICE_BC_ID: 10.0},
    )
    snapshots.append(
        (
            "scenario_a_healthy",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s1_queues,
                counters=make_muni_counters(),
            ),
        )
    )

    # 2. Municipality PT Backlog (10 tokens)
    s2_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 8, SERVICE_WC_ID: 1, SERVICE_BC_ID: 1},
        {SERVICE_PT_ID: 13.0, SERVICE_WC_ID: 5.0, SERVICE_BC_ID: 9.0},
    )
    snapshots.append(
        (
            "scenario_b_pt_pressure",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s2_queues,
                counters=make_muni_counters(),
            ),
        )
    )

    # 3. Municipality Busy In-Progress (6 tokens)
    tok91 = ServingTokenSnapshot(
        token_id=91,
        token_number="PT-91",
        service_id=SERVICE_PT_ID,
        counter_id=1,
        status="SERVING",
        called_at_utc=t_snap - timedelta(minutes=5),
        serving_at_utc=t_snap - timedelta(minutes=5),
    )
    tok92 = ServingTokenSnapshot(
        token_id=92,
        token_number="PT-92",
        service_id=SERVICE_PT_ID,
        counter_id=2,
        status="SERVING",
        called_at_utc=t_snap - timedelta(minutes=3),
        serving_at_utc=t_snap - timedelta(minutes=3),
    )
    s3_counters = make_muni_counters(serving_map={1: tok91, 2: tok92})
    s3_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 5, SERVICE_WC_ID: 1, SERVICE_BC_ID: 0},
        {SERVICE_PT_ID: 12.0, SERVICE_WC_ID: 6.0, SERVICE_BC_ID: 8.0},
    )
    snapshots.append(
        (
            "scenario_b_busy_counters",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s3_queues,
                counters=s3_counters,
            ),
        )
    )

    # 4. Municipality Donor Constraint (13 tokens)
    s4_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 8, SERVICE_WC_ID: 4, SERVICE_BC_ID: 1},
        {SERVICE_PT_ID: 13.0, SERVICE_WC_ID: 7.0, SERVICE_BC_ID: 8.0},
    )
    snapshots.append(
        (
            "scenario_c_donor_constraint",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s4_queues,
                counters=make_muni_counters(),
            ),
        )
    )

    # 5. Municipality Temporary Look-Ahead (8 tokens)
    s5_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 6, SERVICE_WC_ID: 1, SERVICE_BC_ID: 1},
        {SERVICE_PT_ID: 12.0, SERVICE_WC_ID: 5.0, SERVICE_BC_ID: 8.0},
    )
    snapshots.append(
        (
            "scenario_d_temporary",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s5_queues,
                counters=make_muni_counters(),
            ),
        )
    )

    # 6. Municipality Counter Failure (4 tokens, C1 UNAVAILABLE)
    s6_counters = make_muni_counters(active_ids={2, 3, 4, 5, 6})
    s6_queues = make_queues(
        muni_services,
        {SERVICE_PT_ID: 2, SERVICE_WC_ID: 1, SERVICE_BC_ID: 1},
        {SERVICE_PT_ID: 9.0, SERVICE_WC_ID: 5.0, SERVICE_BC_ID: 10.0},
    )
    snapshots.append(
        (
            "scenario_e_counter_failure",
            SimulationSnapshot(
                office_id=OFFICE_MUNICIPALITY_ID,
                office_code=OFFICE_MUNICIPALITY_CODE,
                snapshot_time_utc=t_snap,
                services=muni_services,
                queues=s6_queues,
                counters=s6_counters,
            ),
        )
    )

    # 7. RTO Healthy Operation (5 tokens: OT: 3, DR: 2)
    rto_c_dict = {}
    for c in rto_counters:
        rto_c_dict[c.counter_id] = CounterSnapshot(
            counter_id=c.counter_id,
            office_id=OFFICE_RTO_ID,
            name=c.name,
            status="ACTIVE",
            status_changed_at_utc=t_snap,
            assigned_service_id=c.assigned_service_id,
            eligible_service_ids=c.eligible_service_ids,
            current_token=None,
        )
    rto_queues = make_queues(
        rto_services,
        {SERVICE_OT_ID: 3, SERVICE_DR_ID: 2},
        {SERVICE_OT_ID: 5.5, SERVICE_DR_ID: 5.0},
    )
    snapshots.append(
        (
            "scenario_rto_healthy",
            SimulationSnapshot(
                office_id=OFFICE_RTO_ID,
                office_code=OFFICE_RTO_CODE,
                snapshot_time_utc=t_snap,
                services=rto_services,
                queues=rto_queues,
                counters=rto_c_dict,
            ),
        )
    )

    return snapshots


def generate_proof_dataset(base_seed: int = DEFAULT_BASE_SEED) -> pd.DataFrame:
    """
    Executes authoritative discrete-event simulation runs on controlled snapshots
    and produces the proof dataset containing:
    - Exactly the 15 RAW_FEATURE_NAMES in canonical order
    - Ground-truth target_wait_minutes
    - Minimal necessary metadata (token_id, token_number, queue_position, snapshot_time_utc, scenario_id)
    """
    snapshots = build_proof_snapshots()
    rows: List[Dict] = []

    for scen_idx, (scen_id, snap) in enumerate(snapshots):
        sim_seed = base_seed + scen_idx * 101

        # 1. Convert snapshot to isolated SimulationState
        sim_state = SimulationState.from_snapshot(snap)

        # 2. Run authoritative DiscreteEventEngine
        engine = DiscreteEventEngine(horizon_minutes=180.0, seed=sim_seed)
        result = engine.run(sim_state)

        # 3. Associate waiting tokens with ground-truth simulation results
        t_snap = snap.snapshot_time_utc
        waiting_ids_by_service = {
            s_id: [w.token_id for w in q.waiting_tokens]
            for s_id, q in snap.queues.items()
        }

        for s_id, q in snap.queues.items():
            if not q.waiting_tokens:
                continue

            completed_in_s = [
                t for t in result.completed_citizen_ids
                if t in waiting_ids_by_service[s_id]
            ]
            waits_in_s = result.service_waiting_times[s_id]

            for pos, w_tok in enumerate(q.waiting_tokens, 1):
                # Feature Vector: strictly derived from snapshot at t_snap
                feat_vec = compute_raw_features(snap, service_id=s_id, position=pos)
                row_dict = feat_vec.to_dict()

                # Target Ground Truth: actual future wait from t_snap to service start
                token_idx_in_completed = completed_in_s.index(w_tok.token_id)
                sim_wait = waits_in_s[token_idx_in_completed]
                arr_offset = (w_tok.arrival_time_utc - t_snap).total_seconds() / 60.0
                future_wait = sim_wait + arr_offset

                # Non-negative & finite validation
                future_wait = max(0.0, round(float(future_wait), 4))
                if not math.isfinite(future_wait):
                    raise ValueError(f"Non-finite target_wait_minutes for token {w_tok.token_id}")

                row_dict["target_wait_minutes"] = future_wait

                # Necessary metadata
                row_dict["token_id"] = w_tok.token_id
                row_dict["token_number"] = w_tok.token_number
                row_dict["queue_position"] = pos
                row_dict["snapshot_time_utc"] = t_snap.isoformat()
                row_dict["scenario_id"] = scen_id

                rows.append(row_dict)

    df = pd.DataFrame(rows)

    # Ensure canonical column order: 15 raw features -> target -> metadata
    metadata_cols = ["token_id", "token_number", "queue_position", "snapshot_time_utc", "scenario_id"]
    ordered_cols = list(RAW_FEATURE_NAMES) + ["target_wait_minutes"] + metadata_cols
    df = df[ordered_cols]

    return df


def save_proof_dataset(
    df: pd.DataFrame,
    output_dir: str = DEFAULT_PROOF_OUTPUT_DIR,
) -> Tuple[str, str]:
    """
    Saves proof dataset to CSV and Parquet formats in output_dir.
    Returns (csv_path, parquet_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "proof_dataset.csv")
    parquet_path = os.path.join(output_dir, "proof_dataset.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    return csv_path, parquet_path


def main() -> None:
    print("=" * 60)
    print("PRAVAAH — Step 2E.1A: ML Dataset Generation Proof")
    print("=" * 60)

    df = generate_proof_dataset(base_seed=DEFAULT_BASE_SEED)
    csv_path, parquet_path = save_proof_dataset(df)

    print(f"Generated {len(df)} rows.")
    print(f"Features: {list(RAW_FEATURE_NAMES)}")
    print(f"Target: target_wait_minutes (min: {df['target_wait_minutes'].min():.4f}, "
          f"mean: {df['target_wait_minutes'].mean():.4f}, "
          f"max: {df['target_wait_minutes'].max():.4f})")
    print(f"Offices: {sorted(df['office_id'].unique().tolist())}")
    print(f"Services: {sorted(df['service_id'].unique().tolist())}")
    print(f"CSV saved to: {csv_path}")
    print(f"Parquet saved to: {parquet_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
