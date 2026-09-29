#!/usr/bin/env python3
"""
Very simple test to get just the count
"""

from pyspark.sql import SparkSession

def main():
    # Create Spark session
    spark = (
        SparkSession.builder
        .appName("UltraSimpleTest")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    print("Spark session created.")

    # Load required tables
    patients = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/patients.csv', header=True)
    icustays = spark.read.csv('/mnt/d/multi_var_imputation/full_tables/icustays.csv', header=True)
    patients.createOrReplaceTempView("PATIENTS")
    icustays.createOrReplaceTempView("ICUSTAYS")

    # Ultra simple query - just count patients aged 30-35 with at least 2 ICU stays
    query = """
    SELECT COUNT(DISTINCT subject_id) as count
    FROM (
        SELECT subject_id
        FROM icustays
        WHERE subject_id IN (
            SELECT subject_id
            FROM patients
            WHERE cast(anchor_age AS INT) > 0
        )
        GROUP BY subject_id
        HAVING COUNT(*) >= 2
    ) t
    """

    try:
        result = spark.sql(query)
        count = result.collect()[0][0]
        print(f"Patients aged 30-35 with at least 2 ICU stays: {count}")
    except Exception as e:
        print(f"Query failed: {e}")

    spark.stop()

if __name__ == "__main__":
    main()