-- Query 1: Cohort Analysis
WITH age_cohorts AS (
    SELECT 
        CASE 
            WHEN age < 18 THEN 'Pediatric'
            WHEN age BETWEEN 18 AND 45 THEN 'Young Adult'
            WHEN age BETWEEN 46 AND 64 THEN 'Middle-Aged'
            ELSE 'Geriatric'
        END AS age_group,
        readmitted_within_30_days
    FROM hospital_data
)
SELECT 
    age_group,
    COUNT(*) AS total_patients,
    SUM(readmitted_within_30_days) AS total_readmissions,
    ROUND(AVG(readmitted_within_30_days) * 100, 2) AS readmission_rate_pct
FROM age_cohorts
GROUP BY age_group
ORDER BY readmission_rate_pct DESC;


-- Query 2: Efficiency Metric using Window Functions
SELECT 
    primary_diagnosis_group,
    age,
    length_of_stay,
    -- Partition by diagnosis to find group average
    ROUND(AVG(length_of_stay) OVER(PARTITION BY primary_diagnosis_group), 2) AS avg_los_for_diagnosis,
    -- Variance calculation
    length_of_stay - AVG(length_of_stay) OVER(PARTITION BY primary_diagnosis_group) AS deviation_from_avg
FROM hospital_data
ORDER BY deviation_from_avg DESC;

-- Query 3: Identifying Super-Utilizers
SELECT 
    age, 
    gender, 
    primary_diagnosis_group,
    COUNT(*) AS total_visits,
    ROUND(AVG(severity_score), 2) AS avg_severity,
    SUM(length_of_stay) AS total_days_hospitalized
FROM hospital_data
GROUP BY age, gender, primary_diagnosis_group
HAVING COUNT(*) > 5
ORDER BY total_visits DESC;