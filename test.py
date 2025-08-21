#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
import pandas as pd
import os
from pyspark import SparkConf
import pyarrow
# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

conf=SparkConf()
conf.set("spark.executor.memory", "50g")
conf.set("spark.driver.memory", "50g")
conf.set("spark.sql.execution.arrow.enabled", "true")
conf.set("spark.sql.ansi.enabled", "false")
conf.set("spark.driver.maxResultSize", "0")  # Disable driver result size limit
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100000")  # Adjust batch size for Arrow
conf.set("spark.sql.execution.arrow.fallback.enabled", "true")  # Enable fallback to Pandas if Arrow fails

# Read individual CSV files into Spark DataFrames

# admissions = spark.read.csv('./demo_tables/admissions.csv', header=True)
# patients = spark.read.csv('./demo_tables/patients.csv', header=True)
# chartevents = spark.read.csv('./demo_tables/chartevents.csv', header=True)
# icustays = spark.read.csv('./demo_tables/icustays.csv', header=True)
# diagnoses_icd = spark.read.csv('./demo_tables/diagnoses_icd.csv', header=True)
# d_icd_diagnoses = spark.read.csv('./demo_tables/d_icd_diagnoses.csv', header=True)

# Read individual large CSV files into Spark DataFrames
admissions = spark.read.csv('./full_tables/admissions.csv', header=True)
chartevents = spark.read.csv('./full_tables/chartevents.csv', header=True)
patients = spark.read.csv('./full_tables/patients.csv', header=True)
icustays = spark.read.csv('./full_tables/icustays.csv', header=True)
diagnoses_icd = spark.read.csv('./full_tables/diagnoses_icd.csv', header=True)
d_icd_diagnoses = spark.read.csv('./full_tables/d_icd_diagnoses.csv', header=True)




# Register the DataFrame 'data' as a temporary view for use in SQL queries
admissions.createOrReplaceTempView("ADMISSIONS")
chartevents.createOrReplaceTempView("CHARTEVENTS")
patients.createOrReplaceTempView("PATIENTS")
diagnoses_icd.createOrReplaceTempView("DIAGNOSES_ICD")
d_icd_diagnoses.createOrReplaceTempView("D_ICD_DIAGNOSES")
icustays.createOrReplaceTempView("ICUSTAYS")


# In[ ]:


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
SELECT *
FROM (
    SELECT *,
           unix_timestamp(charttime) AS chart_ts,
           MAX(unix_timestamp(charttime)) OVER (PARTITION BY SUBJECT_ID) AS max_chart_ts
    FROM chartevents_filtered
)
WHERE chart_ts >= max_chart_ts - 48*3600
ORDER BY SUBJECT_ID, charttime
"""

result = spark.sql(query)

# Save each SUBJECT_ID as a separate file (e.g., CSV)
subject_ids = [row.subject_id for row in result.select("subject_id").distinct().collect()]
print(f"Number of unique SUBJECT_IDs: {len(subject_ids)}")


output_base_path = "./Readmitted_patients"

for subject_id in subject_ids:
    # print(f"Processing SUBJECT_ID: {subject_id}")

    # Filter the result for the current subject_id
    filtered_df = result.filter(result.subject_id == subject_id)
    # Remove null values in the 'valuenum' column
    filtered_df = filtered_df.filter(filtered_df.valuenum.isNotNull())

    # Ensure the output directory exists
    os.makedirs(output_base_path, exist_ok=True)

    # Define output path for this subject_id
    output_path = f"{output_base_path}/patient_{subject_id}"

    filtered_df.toPandas().to_parquet(f'{output_path}.parquet', index=False)


# Convert to py file from notebook
print(f"Data for {len(subject_ids)} patients has been saved to {output_base_path} in Parquet format.")
spark.stop()

