# QueueLess Government Office

A discrete-event queue simulation and ML prediction engine to minimize wait times at government offices.

## Overview
This backend powers the QueueLess system, providing citizens, employees, and admins with API access to queue statuses, tokens, and advanced wait-time predictions.

## Architecture & Stack
* **Language:** Python 3.14+
* **Framework:** FastAPI
* **Database:** PostgreSQL (with psycopg v3 and SQLAlchemy)
* **Migrations:** Alembic
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Authentication:** JWT with bcrypt password hashing
* **Real-time:** WebSockets for live queue updates

## Backend Setup & Environment Configuration
1. Clone the repository.
2. Create a Python virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy `.env.example` to `backend/.env` and update `DATABASE_URL` and `JWT_SECRET`.
   * **Note:** The `backend/.env` file is ignored by Git to protect secrets.

## Database & Migrations
1. Ensure PostgreSQL is running.
2. From the `backend/` directory, run migrations to setup the database:
   ```bash
   alembic upgrade head
   ```

## Running the Application
From the `backend/` directory, start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The Swagger UI API documentation will be available at: `http://localhost:8000/docs`.

## Testing
From the `backend/` directory, run the test suite:
```bash
python -m pytest -q
```

## Authentication
The application uses JWT (JSON Web Tokens) for authentication. 
Roles include `citizen`, `employee`, and `admin`. Most endpoints require a valid Bearer token.

## WebSockets
Live queue updates are broadcasted via authenticated WebSockets at:
`/ws/queues/{office_id}/{service_id}`

## Machine Learning
The project includes a Random Forest model located in `ml/models/random_forest_wait_time_pipeline.joblib`. This model is automatically loaded by the backend to predict wait times when a citizen joins a queue.
