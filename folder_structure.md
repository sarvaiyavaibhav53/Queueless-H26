# Project Folder Structure

```text
queueless-government-office/
├── backend
│   ├── alembic
│   │   ├── versions
│   │   │   ├── 7dd9ea24a3c9_add_auth_and_users.py
│   │   │   ├── af098ec6acf6_add_queue_domain_models.py
│   │   │   └── e35bf8ac3b90_initial_office_model.py
│   │   ├── env.py
│   │   ├── README
│   │   └── script.py.mako
│   ├── app
│   │   ├── api
│   │   │   ├── analytics.py
│   │   │   ├── auth.py
│   │   │   ├── deps.py
│   │   │   ├── offices.py
│   │   │   ├── predictions.py
│   │   │   ├── queues.py
│   │   │   ├── services.py
│   │   │   ├── tokens.py
│   │   │   └── ws.py
│   │   ├── core
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── db
│   │   │   └── database.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── office.py
│   │   │   ├── queue_counter.py
│   │   │   ├── service.py
│   │   │   ├── token.py
│   │   │   └── user.py
│   │   ├── schemas
│   │   │   ├── analytics.py
│   │   │   ├── auth.py
│   │   │   ├── office.py
│   │   │   ├── prediction.py
│   │   │   ├── queue.py
│   │   │   └── service.py
│   │   ├── services
│   │   │   ├── prediction_service.py
│   │   │   └── queue_service.py
│   │   ├── main.py
│   │   └── ws_manager.py
│   ├── tests
│   │   ├── conftest.py
│   │   ├── run_tests.py
│   │   ├── test_admin_analytics.py
│   │   ├── test_api.py
│   │   ├── test_api_queue.py
│   │   ├── test_auth.py
│   │   ├── test_authorization.py
│   │   ├── test_citizen_flow.py
│   │   ├── test_db.py
│   │   ├── test_e2e_backend.py
│   │   ├── test_employee_flow.py
│   │   ├── test_hardening.py
│   │   ├── test_phase3_ml_integration.py
│   │   ├── test_queue_engine.py
│   │   └── test_websocket_queue.py
│   ├── .env
│   └── alembic.ini
├── docs
│   ├── backend_cleanup_github_readiness_report.md
│   ├── backend_manual_api_test_report.md
│   ├── final_backend_master_test_report.md
│   ├── ml_prediction_api_test_report.md
│   ├── postgresql_database_setup_test_report.md
│   ├── queue_engine_phase1_test_report.md
│   ├── queue_engine_phase2_api_test_report.md
│   ├── queue_engine_phase2_hardening_test_report.md
│   ├── queue_engine_phase3_ml_integration_test_report.md
│   ├── queue_engine_phase4_test_count_investigation.md
│   ├── queue_engine_phase4_test_hardening_report.md
│   ├── queue_engine_phase4_websocket_test_report.md
│   ├── queue_engine_phase5_citizen_flow_backend_test_report.md
│   ├── queue_engine_phase6_employee_flow_backend_test_report.md
│   ├── queue_engine_phase7_admin_analytics_backend_test_report.md
│   ├── queue_engine_phase8_backend_hardening_test_report.md
│   ├── queue_engine_phase9_authentication_test_report.md
│   └── QueueLess_Government_Office_Full_Project_Plan.docx
├── frontend
├── ml
│   ├── config
│   │   └── simulation_config.yaml
│   ├── data
│   │   └── synthetic
│   │       ├── analysis
│   │       ├── audits
│   │       │   └── SYNTHETIC_TARGET_DEPENDENCY_AUDIT.md
│   │       ├── features
│   │       │   ├── FEATURE_ENGINEERING_REPORT.md
│   │       │   ├── queue_features.csv
│   │       │   ├── queue_features.parquet
│   │       │   ├── queue_features_modeling.csv
│   │       │   └── queue_features_modeling.parquet
│   │       ├── frozen
│   │       │   ├── checksums.sha256
│   │       │   ├── dataset_manifest.json
│   │       │   ├── FREEZE_REPORT.md
│   │       │   ├── generation_metadata.json
│   │       │   ├── queue_events_full.csv
│   │       │   └── queue_events_full.parquet
│   │       ├── full
│   │       │   ├── generation_metadata.json
│   │       │   ├── queue_events_full.csv
│   │       │   └── queue_events_full.parquet
│   │       ├── splits
│   │       │   ├── BASELINE_REPORT.md
│   │       │   ├── split_metadata.json
│   │       │   ├── test.csv
│   │       │   ├── test.parquet
│   │       │   ├── train.csv
│   │       │   ├── train.parquet
│   │       │   ├── validation.csv
│   │       │   └── validation.parquet
│   │       ├── test
│   │       │   ├── analysis
│   │       │   │   ├── arrivals_by_hour.png
│   │       │   │   ├── avg_wait_by_hour.png
│   │       │   │   ├── avg_wait_by_office.png
│   │       │   │   ├── avg_wait_by_service.png
│   │       │   │   ├── queue_length_time.png
│   │       │   │   ├── service_time_dist.png
│   │       │   │   ├── summary_statistics_test.txt
│   │       │   │   └── wait_time_dist.png
│   │       │   ├── queue_events_test.csv
│   │       │   └── queue_events_test.parquet
│   │       └── validation
│   │           ├── robustness_by_active_counters.csv
│   │           ├── robustness_by_hour.csv
│   │           ├── robustness_by_office.csv
│   │           ├── robustness_by_people_ahead.csv
│   │           ├── robustness_by_queue_group.csv
│   │           ├── robustness_by_service.csv
│   │           ├── ROBUSTNESS_VALIDATION_REPORT.md
│   │           └── top_prediction_errors.csv
│   ├── models
│   │   ├── random_forest_metadata.json
│   │   ├── RANDOM_FOREST_REPORT.md
│   │   └── random_forest_wait_time_pipeline.joblib
│   ├── notebooks
│   ├── src
│   │   ├── __init__.py
│   │   ├── analyze_data.py
│   │   ├── arrival_process.py
│   │   ├── audit_simulator.py
│   │   ├── baseline.py
│   │   ├── counter_manager.py
│   │   ├── event_engine.py
│   │   ├── generate_full.py
│   │   ├── queue_engine.py
│   │   ├── robustness_validation.py
│   │   ├── service_process.py
│   │   ├── simulate.py
│   │   ├── train_random_forest.py
│   │   └── validate_data.py
│   ├── feature_engineering.py
│   ├── freeze_dataset.py
│   ├── README.md
│   └── requirements.txt
├── .env.example
├── .gitignore
├── gen_tree.py
├── README.md
└── requirements.txt
```
