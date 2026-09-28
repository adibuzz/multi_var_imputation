#!/usr/bin/env python3
"""
Create one Parquet file per patient with measurements from the last 24 hours.
- Rows: time points (CHARTTIME)
- Columns: measurement types (ITEMID)
- Values: measurement values (VALUE_NUM)
"""

import os
from typing import Any, List

from pyspark.sql import SparkSession

def main() -> None:
    # Create Spark session with configuration similar to test.py
    spark = (
        SparkSession.builder
        .appName("PatientMeasurementsLast24h")
        .config("spark.executor.memory", "50g")
        .config("spark.driver.memory", "50g")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )

    print("Spark session created.")

    # Define the ITEMIDs (same as in test.py)
    ITEMIDS: list[Any] = [220546, 224828, 220644, 220235, 225624, 229761,
                   220363, 220422, 220467, 220339, 224696, 225170, 227466, 220050,
                   220051, 220052, 228386, 220074, 220367, 220045, 225651, 226754,
                   226755, 227015, 227467, 220451, 224697, 225168, 220210,
                   220227, 223761, 225690, 220650]

    print(f"Using {len(ITEMIDS)} measurement ITEMIDs.")

    # Load individual CSV files into Spark DataFrames (same paths as test.py)
    print("Loading data...")
    admissions = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/admissions.csv', header=True)
    chartevents = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/chartevents.csv', header=True)
    patients = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/patients.csv', header=True)
    icustays = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/icustays.csv', header=True)
    diagnoses_icd = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/diagnoses_icd.csv', header=True)
    d_icd_diagnoses = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/d_icd_diagnoses.csv', header=True)

    # Register the DataFrames as temporary views for use in SQL queries
    admissions.createOrReplaceTempView("ADMISSIONS")
    chartevents.createOrReplaceTempView("CHARTEVENTS")
    patients.createOrReplaceTempView("PATIENTS")
    diagnoses_icd.createOrReplaceTempView("DIAGNOSES_ICD")
    d_icd_diagnoses.createOrReplaceTempView("D_ICD_DIAGNOSES")
    icustays.createOrReplaceTempView("ICUSTAYS")

    # Execute the exact same query as in test.py
    print("Executing the query from test.py...")
    query = """
    WITH d_calc AS (
        SELECT
            SUBJECT_ID,
            CAST(intime AS timestamp) AS intime,
            CAST(outtime AS timestamp) AS outtime,
            row_number() OVER (PARTITION BY SUBJECT_ID ORDER BY CAST(intime AS timestamp) ASC) AS record_seq,
            HADM_ID,
            STAY_ID,
            LAG(CAST(outtime AS timestamp), 1) OVER (
                PARTITION BY SUBJECT_ID
                ORDER BY CAST(intime AS timestamp)
            ) AS previous_outtime
        FROM ICUSTAYS
        WHERE SUBJECT_ID IN (
            SELECT SUBJECT_ID
            FROM PATIENTS
            WHERE CAST(anchor_age AS INT) > 65
        )
    ),
    d_days AS (
        SELECT
            SUBJECT_ID,
            HADM_ID,
            STAY_ID,
            CAST(intime AS double) - CAST(previous_outtime AS double) AS Duration,
            record_seq
        FROM d_calc
        WHERE previous_outtime IS NOT NULL
    ),
    d_days_filtered_subject_id AS (
        SELECT
            SUBJECT_ID AS SUB_ID,
            HADM_ID AS H_ID,
            STAY_ID AS S_ID,
            Duration / 86400 AS LoS,
            record_seq
        FROM d_days
        WHERE SUBJECT_ID IN (SELECT SUBJECT_ID FROM d_days WHERE record_seq = 2)
          AND record_seq <= 2
        ORDER BY SUBJECT_ID
    ),
    HADM_IDs AS (
        SELECT *
        FROM d_days_filtered_subject_id
        WHERE record_seq = 1
    ),
    chartevents_filtered AS (
        SELECT *
        FROM chartevents
        WHERE ITEMID IN (220546, 224828, 220644, 220235, 225624, 229761,
              220363, 220422, 220467, 220339, 224696, 225170, 227466, 220050,
                 220051, 220052, 228386, 220074, 220367, 220045, 225651, 226754,
                        226755, 227015, 227467, 220451, 224697, 225168, 220210,
                            220227, 223761, 225690, 220650)
            AND SUBJECT_ID IN (SELECT SUB_ID FROM d_days_filtered_subject_id)
    )
    SELECT SUBJECT_ID, charttime, itemid, valuenum
    FROM (
        SELECT *,
               unix_timestamp(charttime) AS chart_ts,
               MAX(unix_timestamp(charttime)) OVER (PARTITION BY SUBJECT_ID) AS max_chart_ts
        FROM chartevents_filtered
    )
    WHERE chart_ts >= max_chart_ts - 24*3600
    ORDER BY SUBJECT_ID, charttime
    """

    result = spark.sql(query)

    print(f"Query executed. Found {result.count()} measurement records.")

    from pyspark.sql import functions as f

    # FIXED: Efficient distributed writing instead of collect-loop
    output_path = "./Readmitted_patients"

    # Write directly as partitioned Parquet - much more efficient and scalable
    result.filter(f.col("valuenum").isNotNull()) \
        .write \
        .partitionBy("SUBJECT_ID") \
        .mode("overwrite") \
        .parquet(output_path)

    print(f"Data written to {output_path} in partitioned Parquet format")

    # Optional: Show some statistics
    print(f"Total patients processed: {result.select('SUBJECT_ID').distinct().count()}")

    # Properly stop the Spark session to free up resources
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()