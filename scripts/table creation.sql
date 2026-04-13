DROP TABLE hospital_data;

CREATE TABLE hospital_data (
    age INT,
    gender VARCHAR(20),
    insurance_type VARCHAR(50),
    socioeconomic_risk_score INT,
    previous_admissions_6m INT,
    previous_readmissions_1y INT,
    time_since_last_discharge INT,
    length_of_stay INT,
    admission_type VARCHAR(50),
    primary_diagnosis_group VARCHAR(100),
    comorbidity_index INT,
    chronic_disease_count INT,
    icu_stay_flag INT,
    severity_score INT,
    hba1c_level DECIMAL(5,2),
    creatinine_level DECIMAL(5,2),
    hemoglobin_level DECIMAL(5,2),
    average_systolic_bp INT,
    number_of_medications INT,
    medication_change_count INT,
    high_risk_medication_flag INT,
    followup_appointment_scheduled INT,
    discharge_disposition VARCHAR(100),
    medication_adherence_score DECIMAL(5,2),
    readmitted_within_30_days INT
);

COPY hospital_data 
FROM 'C:\Users\Public\cvs_project.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',');

SELECT * FROM hospital_data LIMIT 10;