\connect swasthya_db;

-- =============================================
-- 1. Demand Forecasting
-- Stores predictions from the Demand Forecast Agent
-- =============================================
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    forecast_date DATE NOT NULL,
    predicted_volume FLOAT NOT NULL,
    lower_bound FLOAT, -- Confidence interval lower
    upper_bound FLOAT, -- Confidence interval upper
    department VARCHAR(50) DEFAULT 'GENERAL', -- e.g., ER, ICU
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) -- Links to MLflow run ID
);

-- =============================================
-- 2. Staffing
-- Core staff roster
-- =============================================
CREATE TABLE IF NOT EXISTS staff (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL, -- e.g., 'Doctor', 'Nurse', 'Surgeon'
    qualification_level VARCHAR(50), -- e.g., 'Senior', 'Junior', 'Specialist'
    shift_preference VARCHAR(20) DEFAULT 'Day'
);

-- Generated schedules from the Staff Scheduling Agent
CREATE TABLE IF NOT EXISTS staff_schedules (
    id SERIAL PRIMARY KEY,
    staff_id INT REFERENCES staff(id),
    shift_start TIMESTAMP NOT NULL,
    shift_end TIMESTAMP NOT NULL,
    assigned_department VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 3. Triage & Acuity
-- Logs from the Triage Agent (XGBoost + NLP)
-- =============================================
CREATE TABLE IF NOT EXISTS triage_decisions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50), -- Could be hash of Aadhaar in future
    symptoms_text TEXT,
    vitals_json JSONB, -- Stores {heart_rate: 80, bp: "120/80", ...}
    acuity_score INT NOT NULL, -- 1 (Critical) to 5 (Non-urgent)
    predicted_risk_score FLOAT, -- Probability output from XGBoost
    is_override BOOLEAN DEFAULT FALSE, -- If human overrode the AI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 4. ER & OR Scheduling
-- =============================================
CREATE TABLE IF NOT EXISTS er_queue (
    id SERIAL PRIMARY KEY,
    triage_id INT REFERENCES triage_decisions(id),
    patient_id VARCHAR(50),
    arrival_time TIMESTAMP NOT NULL,
    priority_score INT, -- Calculated based on acuity + wait time
    status VARCHAR(20) DEFAULT 'WAITING' -- WAITING, TREATING, DISCHARGED, ADMITTED
);

CREATE TABLE IF NOT EXISTS or_schedules (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50),
    surgery_type VARCHAR(100),
    operating_room_id VARCHAR(20),
    scheduled_start TIMESTAMP NOT NULL,
    predicted_duration_minutes FLOAT, -- Predicted by XGBoost model
    actual_duration_minutes FLOAT,    -- For retraining
    status VARCHAR(20) DEFAULT 'SCHEDULED'
);

-- =============================================
-- 5. Inpatient & Discharge
-- =============================================
CREATE TABLE IF NOT EXISTS inpatients (
    admission_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admit_date TIMESTAMP NOT NULL,
    ward_location VARCHAR(50),
    attending_doctor_id INT REFERENCES staff(id),
    current_status VARCHAR(50) DEFAULT 'STABLE'
);

CREATE TABLE IF NOT EXISTS discharge_recommendations (
    id SERIAL PRIMARY KEY,
    admission_id INT REFERENCES inpatients(admission_id),
    readiness_score FLOAT, -- 0.0 to 1.0 from Discharge Agent
    recommended_action VARCHAR(50), -- 'DISCHARGE', 'MONITOR', 'RETAIN'
    notes TEXT, -- Explanation or rule hit (e.g., "Temp stable > 24h")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 6. Seed Data (For Development/Demo)
-- =============================================

INSERT INTO staff (name, role, qualification_level) VALUES 
('Dr. Anjali Rao', 'Doctor', 'Senior'),
('Nurse Rajesh', 'Nurse', 'Junior'),
('Dr. Smitha', 'Surgeon', 'Specialist');

INSERT INTO inpatients (patient_id, admit_date, ward_location, attending_doctor_id) VALUES
('PAT-1001', NOW() - INTERVAL '5 days', 'Ward A', 1),
('PAT-1002', NOW() - INTERVAL '2 days', 'ICU', 1);

-- Initial Dummy Forecast
INSERT INTO forecasts (forecast_date, predicted_volume, department) VALUES
(CURRENT_DATE + INTERVAL '1 day', 120.5, 'ER');
