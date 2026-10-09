"""
PRAVAAH — Idempotent Database Seed Script.

Initializes or updates synthetic demonstration data:
- Offices: Municipality (ID 1), RTO (ID 2)
- Services: PT (1), WC (2), BC (3), OT (4), DR (5)
- Physical Counters & Canonical Home Assignments: C1..C6 (1..6), R1..R3 (201..203)
- Counter Eligibility Mappings (CounterService)
- Synthetic Demo Users: admin@gov.in, employee@gov.in, citizen@example.com

Guarantees:
- Idempotent: repeated runs produce the same state without duplicate records.
- Explicit database target reporting.
- Synthetic demo data only; zero sensitive credentials.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend package is in python search path
repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.database import SessionLocal, engine
from app.models.base import Base
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.user import User, UserRole


def seed_database():
    print("=" * 60)
    print("PRAVAAH Database Seed Script")
    print("=" * 60)
    print(f"Target Database URL: {settings.DATABASE_URL}")

    # 1. Create tables if not present
    Base.metadata.create_all(bind=engine)
    print("[+] Verified / created database schema tables.")

    db = SessionLocal()
    try:
        now_utc = datetime.now(timezone.utc)

        # 2. Seed Offices
        offices_data = [
            {"id": 1, "code": "MUNI", "name": "City Municipal Corporation", "counters": 6, "multiplier": 1.0},
            {"id": 2, "code": "RTO", "name": "Regional Transport Office", "counters": 3, "multiplier": 1.0},
        ]
        for o_info in offices_data:
            existing = db.query(Office).filter(Office.id == o_info["id"]).first()
            if not existing:
                db.add(Office(
                    id=o_info["id"],
                    code=o_info["code"],
                    name=o_info["name"],
                    counters=o_info["counters"],
                    demand_multiplier=o_info["multiplier"],
                ))
                print(f"[+] Created Office {o_info['code']} (ID {o_info['id']})")
            else:
                existing.code = o_info["code"]
                existing.name = o_info["name"]
                existing.counters = o_info["counters"]
                existing.demand_multiplier = o_info["multiplier"]
                print(f"[*] Verified Office {o_info['code']} (ID {o_info['id']})")
        db.flush()

        # 3. Seed Services
        services_data = [
            {"id": 1, "code": "PT", "name": "Property Tax Assessment & Payment", "mean": 10.0, "std": 3.0},
            {"id": 2, "code": "WC", "name": "Water Connection & Metering", "mean": 15.0, "std": 5.0},
            {"id": 3, "code": "BC", "name": "Birth & Death Certificate Services", "mean": 7.0, "std": 2.0},
            {"id": 4, "code": "OT", "name": "Vehicle Ownership Transfer", "mean": 12.0, "std": 4.0},
            {"id": 5, "code": "DR", "name": "Driving License Renewal", "mean": 8.0, "std": 2.5},
        ]
        for s_info in services_data:
            existing = db.query(Service).filter(Service.id == s_info["id"]).first()
            if not existing:
                db.add(Service(
                    id=s_info["id"],
                    code=s_info["code"],
                    name=s_info["name"],
                    mean_minutes=s_info["mean"],
                    std_minutes=s_info["std"],
                ))
                print(f"[+] Created Service {s_info['code']} (ID {s_info['id']})")
            else:
                existing.code = s_info["code"]
                existing.name = s_info["name"]
                existing.mean_minutes = s_info["mean"]
                existing.std_minutes = s_info["std"]
                print(f"[*] Verified Service {s_info['code']} (ID {s_info['id']})")
        db.flush()

        # 4. Seed Counters (Canonical Home Assignments)
        counters_data = [
            # Municipality (Office 1)
            {"id": 1, "office_id": 1, "name": "Counter 1", "home_service_id": 1, "eligibility": [1]},
            {"id": 2, "office_id": 1, "name": "Counter 2", "home_service_id": 1, "eligibility": [1]},
            {"id": 3, "office_id": 1, "name": "Counter 3", "home_service_id": 2, "eligibility": [1, 2]},
            {"id": 4, "office_id": 1, "name": "Counter 4", "home_service_id": 2, "eligibility": [1, 2]},
            {"id": 5, "office_id": 1, "name": "Counter 5", "home_service_id": 3, "eligibility": [2, 3]},
            {"id": 6, "office_id": 1, "name": "Counter 6", "home_service_id": 3, "eligibility": [2, 3]},
            # RTO (Office 2)
            {"id": 201, "office_id": 2, "name": "RTO Counter 1", "home_service_id": 4, "eligibility": [4]},
            {"id": 202, "office_id": 2, "name": "RTO Counter 2", "home_service_id": 4, "eligibility": [4, 5]},
            {"id": 203, "office_id": 2, "name": "RTO Counter 3", "home_service_id": 5, "eligibility": [4, 5]},
        ]
        for c_info in counters_data:
            existing = db.query(Counter).filter(Counter.id == c_info["id"]).first()
            if not existing:
                counter = Counter(
                    id=c_info["id"],
                    office_id=c_info["office_id"],
                    name=c_info["name"],
                    status=CounterStatus.ACTIVE,
                    assigned_service_id=c_info["home_service_id"],
                    status_changed_at=now_utc,
                )
                db.add(counter)
                print(f"[+] Created Counter {c_info['name']} (ID {c_info['id']}) -> Home S{c_info['home_service_id']}")
            else:
                existing.office_id = c_info["office_id"]
                existing.name = c_info["name"]
                existing.status = CounterStatus.ACTIVE
                existing.assigned_service_id = c_info["home_service_id"]
                existing.status_changed_at = now_utc
                print(f"[*] Verified Counter {c_info['name']} (ID {c_info['id']})")

            # Eligibility mappings
            db.query(CounterService).filter(CounterService.counter_id == c_info["id"]).delete()
            for el_s_id in c_info["eligibility"]:
                db.add(CounterService(counter_id=c_info["id"], service_id=el_s_id))
        db.flush()

        # 5. Seed Synthetic Users
        users_data = [
            {"email": "admin@gov.in", "role": UserRole.admin, "password": "Admin@123"},
            {"email": "employee@gov.in", "role": UserRole.employee, "password": "Employee@123"},
            {"email": "citizen@example.com", "role": UserRole.citizen, "password": "Citizen@123"},
        ]
        for u_info in users_data:
            existing = db.query(User).filter(User.email == u_info["email"]).first()
            if not existing:
                db.add(User(
                    email=u_info["email"],
                    password_hash=get_password_hash(u_info["password"]),
                    role=u_info["role"],
                    is_active=True,
                ))
                print(f"[+] Created Synthetic User '{u_info['email']}' ({u_info['role'].value})")
            else:
                existing.role = u_info["role"]
                existing.is_active = True
                print(f"[*] Verified Synthetic User '{u_info['email']}' ({u_info['role'].value})")

        db.commit()
        print("=" * 60)
        print("Database seed completed successfully.")
        print("Demo Credentials:")
        print("  - Admin:    admin@gov.in / Admin@123")
        print("  - Employee: employee@gov.in / Employee@123")
        print("  - Citizen:  citizen@example.com / Citizen@123")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"[-] Database seed error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
