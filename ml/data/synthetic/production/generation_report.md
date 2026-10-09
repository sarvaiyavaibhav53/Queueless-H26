# PRAVAAH — Step 2E.1C Production Dataset Generation Report

**Generated at UTC**: `2026-10-08T12:18:53.405564+00:00`  
**Generator Version**: `2E.1C_Production_v1`  
**Base Seed**: `1000000`  

---

## 1. Executive Summary & Production Volume

* **Total Independent Trajectories**: `1500` (Target: $\ge 1,000$)
* **Total Candidate Observations**: `32,802`
* **Valid Retained Supervised Rows**: `32,802` (Target: 30,000–40,000)
* **Censored Observations**: `0` (0.0%)
* **Per-Trajectory Retained Rows**: Mean `21.99` | Median `20.0` | Min `1` | Max `40`
* **Execution Runtime**: Wall clock `10.64 s` | Sim compute `10.18 s` (`6.79 ms`/trajectory)

---

## 2. Coverage Breakdowns

### A. Office Coverage
| Office ID | Description | Supervised Rows | Percentage |
| :--- | :--- | ---: | ---: |
| `MUNI` | Municipality | 23,011 | 70.15% |
| `RTO` | Regional Transport Office | 9,791 | 29.85% |

### B. Service Coverage
| Service ID | Service Name | Office | Supervised Rows | Percentage |
| :--- | :--- | :--- | ---: | ---: |
| `PT` | Property Tax | `MUNI` | 10,927 | 33.31% |
| `BC` | Birth Certificate | `MUNI` | 6,052 | 18.45% |
| `WC` | Water Service | `MUNI` | 6,032 | 18.39% |
| `OT` | Ownership Transfer | `RTO` | 5,767 | 17.58% |
| `DR` | Duplicate RC | `RTO` | 4,024 | 12.27% |

### C. Operational Regime Breakdown
| Regime | Description | Supervised Rows | Percentage |
| :--- | :--- | ---: | ---: |
| `HEALTHY` | Traffic intensity rho < 0.85 (stable operation) | 11,525 | 35.14% |
| `MODERATE` | Traffic intensity 0.85 <= rho < 1.0 (watch zone) | 8,863 | 27.02% |
| `HIGH_LOAD` | Traffic intensity around/above capacity (rho ~ 1.05 - 1.20) | 5,398 | 16.46% |
| `STRESSED` | Sustained queue buildup (rho >= 1.25 or initial backlog) | 2,963 | 9.03% |
| `FAILURE` | Physical counter outage and recovery events | 2,120 | 6.46% |
| `INTERVENTION` | Temporary counter reassignment with switching delay | 1,933 | 5.89% |

### D. Observation Type Breakdown
| Observation Type | Description | Supervised Rows | Percentage |
| :--- | :--- | ---: | ---: |
| `PERIODIC` | 10-minute snapshot (Head, Middle, Tail positions) | 20,885 | 63.67% |
| `JOIN` | Point-in-time state as a citizen joins the queue | 11,917 | 36.33% |

---

## 3. Target Distribution (`target_wait_minutes`)

* **Count**: `32,802`
* **Min**: `0.0 min`
* **Median (P50)**: `7.0837 min`
* **Mean**: `11.6538 min`
* **P75**: `15.1092 min`
* **P90**: `28.3102 min`
* **P95**: `38.9378 min`
* **Max**: `138.3594 min`
* **Standard Deviation**: `13.3113 min`
* **Zero-Wait Observations**: `1` (0.0%)

---

## 4. JOIN vs PERIODIC Detailed Comparison

| Metric | `JOIN` (Arrival Instant) | `PERIODIC` (10m Interval) |
| :--- | ---: | ---: |
| Row Count | 11,917 | 20,885 |
| Target Mean | 13.949 min | 10.3442 min |
| Target Median | 8.7434 min | 6.2325 min |
| Target P90 | 33.856 min | 25.0 min |
| Mean Queue Position | 2.85 | 2.14 |

---

## 5. Duplication & Autocorrelation Audit

* **Exact 15-Feature Duplicates**: `3` (0.01%)
* **Total Unique Tokens Sampled**: `20,499`
* **Tokens Sampled Multiple Times Within a Trajectory**: `9,439`
* **Near-Duplicate Rows**: `10,920` (33.29%)

---

## 6. Censoring Analysis

| Regime | Candidate Obs | Censored Obs | Censoring % |
| :--- | ---: | ---: | ---: |
| `FAILURE` | 2,120 | 0 | 0.0% |
| `HEALTHY` | 11,525 | 0 | 0.0% |
| `HIGH_LOAD` | 5,398 | 0 | 0.0% |
| `INTERVENTION` | 1,933 | 0 | 0.0% |
| `MODERATE` | 8,863 | 0 | 0.0% |
| `STRESSED` | 2,963 | 0 | 0.0% |

---

## 7. Data Leakage & Integrity Verification

* **Leakage Audit Status**: `PASS`
* **Future Arrival Inspection**: 0% (Strictly prohibited)
* **Future Failure / Intervention Inspection**: 0% (Strictly prohibited)
* **Target-Derived Input Features**: 0% (Strictly prohibited)
* **Canonical 15-Feature Contract Verified**: `PASS`
