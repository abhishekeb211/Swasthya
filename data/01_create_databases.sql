-- =============================================
-- Swasthya Database Initialization
-- Creates required databases and users
-- =============================================

-- Check if database exists, if not create it (idempotent check usually handled by PG init, 
-- but explicit creation helps in some docker setups)
SELECT 'CREATE DATABASE swasthya_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'swasthya_db')\gexec

SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

-- Create a user for mlflow if distinct access control is needed
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
