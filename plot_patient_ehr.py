#!/usr/bin/env python3
"""
Create an enhanced plot for one patient showing measurements over time.
- X-axis: hours before discharge (1, 2, 3, ..., 48)
- Y-axis: measurement values (one line per ITEMID, using item names)
- Measurements visible with markers
- Colorful and easily understandable
- Awaits approval before proceeding
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

def create_itemid_mapping():
    """Create mapping from ITEMID to label using d_items.csv"""
    try:
        df_items = pd.read_csv('demo_tables/d_items.csv')
        # Our ITEMIDs from test.py
        ITEMIDS = [220546, 224828, 220644, 220235, 225624, 229761,
                   220363, 220422, 220467, 220339, 224696, 225170, 227466, 220050,
                   220051, 220052, 228386, 220074, 220367, 220045, 225651, 226754,
                   226755, 227015, 227467, 220451, 224697, 225168, 220210,
                   220227, 223761, 225690, 220650]
        # Create mapping dictionary
        itemid_to_label = dict(zip(
            df_items[df_items['itemid'].isin(ITEMIDS)]['itemid'],
            df_items[df_items['itemid'].isin(ITEMIDS)]['label']
        ))
        return itemid_to_label
    except Exception as e:
        print(f"Warning: Could not load d_items.csv: {e}")
        # Fallback to using ITEMIDs as labels
        ITEMIDS = [220546, 224828, 220644, 220235, 225624, 229761,
                   220363, 220422, 220467, 220339, 224696, 225170, 227466, 220050,
                   220051, 220052, 228386, 220074, 220367, 220045, 225651, 226754,
                   226755, 227015, 227467, 220451, 224697, 225168, 220210,
                   220227, 223761, 225690, 220650]
        return {itemid: str(itemid) for itemid in ITEMIDS}

def main():
    print("Creating enhanced patient measurement plot...")
    print("=" * 50)

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark_conf_dir = os.path.join(os.path.dirname(__file__), "spark_conf")
    os.environ.setdefault("SPARK_CONF_DIR", spark_conf_dir)

    # Create Spark session
    spark = SparkSession.builder \
        .appName("PlotPatientMeasurementsEnhanced") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("✓ Spark session created.")

    # Create ITEMID to label mapping
    print("Loading ITEMID to label mapping from d_items.csv...")
    itemid_to_label = create_itemid_mapping()
    print(f"✓ Loaded mapping for {len(itemid_to_label)} ITEMIDs.")

    # Choose a patient to plot (using the first one from demo data)
    patient_id = 19965802
    parquet_path = f'./Readmitted_patients/SUBJECT_ID={patient_id}'

    # Check if file exists
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        print("Running demo processing first...")
        # Import and run the test processing to generate files
        # from test_create_patient_parquets import main as process_demo
        # process_demo()

    # Read the parquet file using Spark and convert to pandas for plotting
    print(f"Reading parquet file for patient {patient_id}...")
    df_spark = spark.read.parquet(parquet_path)

    # Convert to pandas for easier plotting
    df_pandas = df_spark.toPandas()

    print(f"✓ Patient {patient_id} data loaded: {df_pandas.shape[0]} time points, {df_pandas.shape[1]} columns")
    print(f"Columns: {list(df_pandas.columns)}")

    # Parquet rows are long-form: one measurement per charttime/itemid row.
    time_col = 'charttime'
    itemid_col = 'itemid'
    value_col = 'valuenum'
    df_pandas[time_col] = pd.to_datetime(df_pandas[time_col], errors='coerce')
    df_pandas[itemid_col] = pd.to_numeric(df_pandas[itemid_col], errors='coerce')
    df_pandas[value_col] = pd.to_numeric(df_pandas[value_col], errors='coerce')
    df_pandas = df_pandas.dropna(subset=[time_col, itemid_col])
    if df_pandas.empty:
        raise ValueError("Patient parquet contains no valid charttime/itemid rows")

    measurement_itemids = sorted(df_pandas[itemid_col].unique())
    print(f"✓ Found {len(measurement_itemids)} measurements to plot")

    # Compute hours before the latest charttime in the data.
    discharge_time = df_pandas[time_col].max()
    print(f"Discharge time: {discharge_time}")

    df_pandas['hours_before_discharge'] = (
        (discharge_time - df_pandas[time_col]).dt.total_seconds() / 3600
    )

    print(f"Hours before discharge range: {df_pandas['hours_before_discharge'].min():.2f} to {df_pandas['hours_before_discharge'].max():.2f}")

    # Create the plot
    print("Creating plot...")
    plt.figure(figsize=(16, 10))

    # Define a colorful palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(measurement_itemids)))

    # Plot each measurement as a line with markers
    plotted_any = False
    for i, itemid_value in enumerate(measurement_itemids):
        itemid = int(itemid_value)
        label = itemid_to_label.get(itemid, f"ITEMID {itemid}")

        valid_data = df_pandas.loc[
            df_pandas[itemid_col] == itemid_value,
            ['hours_before_discharge', value_col]
        ].dropna().sort_values('hours_before_discharge')
        if len(valid_data) > 0:
            plt.plot(valid_data['hours_before_discharge'], valid_data[value_col],
                    label=label,
                    color=colors[i],
                    linewidth=2,
                    marker='o',  # Add markers
                    markersize=10,
                    alpha=0.8,
                    markeredgecolor='white',
                    markeredgewidth=0.5)
            plotted_any = True

    if not plotted_any:
        print("Warning: No valid data to plot!")
        spark.stop()
        return

    # Customize the plot
    plt.xlabel('Hours Before Discharge', fontsize=14, fontweight='bold')
    plt.ylabel('Measurement Value', fontsize=14, fontweight='bold')
    plt.title(f'Patient {patient_id}: Measurements Over Time\n' +
              f'({len(measurement_itemids)} analytes tracked, hours before discharge)',
              fontsize=16, fontweight='bold', pad=20)

    # Improve legend - put it outside the plot to avoid overlapping
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=10, ncol=2 if len(measurement_itemids) > 15 else 1,
               frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis to show integer hours (ticks at whole numbers)
    max_hours = int(np.ceil(df_pandas['hours_before_discharge'].max()))
    plt.xticks(range(0, max_hours + 1, max(1, max_hours // 10)))

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot
    plot_filename = f'patient_{patient_id}_measurements_enhanced.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Enhanced plot saved as: {plot_filename}")

    # Also save a PDF version for high quality
    pdf_filename = f'patient_{patient_id}_measurements_enhanced.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"✓ High-quality PDF saved as: {pdf_filename}")

    # Show some statistics
    print(f"\n=== Plot Statistics ===")
    print(f"Patient ID: {patient_id}")
    print(f"Discharge time: {discharge_time}")
    print(f"Time range in data: {df_pandas[time_col].min()} to {df_pandas[time_col].max()}")
    print(f"Hours before discharge range: {df_pandas['hours_before_discharge'].min():.2f} to {df_pandas['hours_before_discharge'].max():.2f}")
    print(f"Number of time points: {len(df_pandas)}")
    print(f"Number of measurements: {len(measurement_itemids)}")

    # Count non-null values for each measurement
    print(f"\nMeasurement completeness (top 10 by completeness):")
    completeness = []
    for itemid_value in measurement_itemids:
        itemid = int(itemid_value)
        label = itemid_to_label.get(itemid, f"ITEMID {itemid}")
        item_data = df_pandas[df_pandas[itemid_col] == itemid_value]
        non_null_count = item_data[value_col].notna().sum()
        total_count = len(item_data)
        pct = (non_null_count / total_count) * 100 if total_count > 0 else 0
        completeness.append((label, non_null_count, total_count, pct))

    # Sort by completeness (descending)
    completeness.sort(key=lambda x: x[2], reverse=True)

    for label, non_null, total, pct in completeness[:10]:
        print(f"  {label}: {non_null}/{total} ({pct:.1f}%)")
    if len(completeness) > 10:
        print(f"  ... and {len(completeness) - 10} more measurements")

    # Clean up
    spark.stop()
    print("\n" + "=" * 50)
    print("Enhanced plot generation completed successfully!")
    print(f"Files created:")
    print(f"  - {plot_filename} (PNG, 300 DPI)")
    print(f"  - {pdf_filename} (PDF, vector)")
    print("=" * 50)

if __name__ == "__main__":
    main()