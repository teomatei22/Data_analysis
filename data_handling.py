import re
import time
import memory_profiler
import pandas as pd
import dask.dataframe as dd

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

import matplotlib.pyplot as plt

# Dummy sentiment scorer
def process_text(text):
    if text is None or pd.isna(text):
        return (0, 0)
    # Tokenize words (alphanumeric)
    words = re.findall(r'\w+', text)
    word_count = len(words)
    # Fake sentiment: sum of Unicode values of characters mod 100
    sentiment_score = sum(ord(c) for c in text) % 100
    return word_count, sentiment_score

# Measure time and memory usage
def measure_memory_and_time(func, *args, **kwargs):
    mem_before = memory_profiler.memory_usage()[0]  # MB
    t_start = time.time()
    result = func(*args, **kwargs)
    t_end = time.time()
    mem_after = memory_profiler.memory_usage()[0]  # MB

    elapsed_time = t_end - t_start
    mem_usage = mem_after - mem_before  # MB

    return result, elapsed_time, mem_usage


if __name__ == "__main__":

    CSV_PATH = 'Books_rating.csv'

    bookfile = open(CSV_PATH, 'r')
    lines=  [bookfile.readline() for i in range(20000)]
    bookout = open("books.csv", 'w')
    for line in lines:
        bookout.write(line)

    CSV_PATH = "books.csv"
    print("Pandas Loading")
    (df_pandas, time_pandas_load, mem_pandas_load) = measure_memory_and_time(
        pd.read_csv,
        CSV_PATH,
        encoding='utf-8',
        dtype={'Id': 'string'}

    )
    print(f"Pandas DataFrame Loaded. Shape: {df_pandas.shape}")
    print(f"  Load Time (s):       {time_pandas_load:.4f}")
    print(f"  Memory Delta (MB):   {mem_pandas_load:.4f}\n")

    if "review/text" in df_pandas.columns:
        df_pandas.rename(columns={"review/text": "review_text"}, inplace=True)
    df_pandas.fillna({"review_text": ""})
    # Processing with Pandas
    def pandas_processing(df):
        df['processed'] = df['review_text'].apply(process_text)
        df['word_count'] = df['processed'].apply(lambda x: x[0])
        df['sentiment_score'] = df['processed'].apply(lambda x: x[1])
        return df

    print("Pandas Processing")
    (df_pandas_processed, time_pandas_proc, mem_pandas_proc) = measure_memory_and_time(
        pandas_processing,
        df_pandas
    )
    print(f"Pandas processing completed.")
    print(f"  Processing Time (s): {time_pandas_proc:.4f}")
    print(f"  Memory Delta (MB):   {mem_pandas_proc:.4f}\n")


    print("Dask Loading")
    (df_dask, time_dask_load, mem_dask_load) = measure_memory_and_time(
        dd.read_csv,
        CSV_PATH,
        encoding='utf-8',
        blocksize="16MB",
        dtype={'Id': 'string'}
    )
    print(f"Dask DataFrame Loaded.")
    print(f"  Load Time (s):       {time_dask_load:.4f}")
    print(f"  Memory Delta (MB):   {mem_dask_load:.4f}\n")

    f_dask = dd.read_csv(CSV_PATH, encoding='utf-8', blocksize="16MB")

    df_dask = df_dask.rename(columns={"review/text": "review_text"})

    df_dask = df_dask.fillna({"review_text": ""})

    def dask_process_partition(df_partition):

        df_partition['processed'] = df_partition['review_text'].apply(process_text)
        df_partition[['word_count', 'sentiment_score']] = df_partition['processed'].apply(
            lambda x: pd.Series(x)
        )
        return df_partition

    def dask_processing(dask_df):

        df_transformed = dask_df.map_partitions(dask_process_partition)
        return df_transformed.compute()  # triggers all computations

    print("Dask Processing")
    (df_dask_processed, time_dask_proc, mem_dask_proc) = measure_memory_and_time(
        dask_processing,
        df_dask
    )
    print("Dask processing completed.")
    print(f"  Processing Time (s): {time_dask_proc:.4f}")
    print(f"  Memory Delta (MB):   {mem_dask_proc:.4f}")
    print(f"Dask DataFrame result shape: {df_dask_processed.shape}\n")


    print("PySpark Loading")
    spark = SparkSession.builder \
                        .appName("BigSalesDataProcessing") \
                        .getOrCreate()

    def spark_load():
        df_spark_local = spark.read.option("header", "true") \
                                   .csv(CSV_PATH, inferSchema=True)
        # Just do a count so we confirm the load
        row_count_local = df_spark_local.count()
        return df_spark_local, row_count_local

    (spark_tuple, time_spark_load, mem_spark_load) = measure_memory_and_time(spark_load)
    df_spark, row_count = spark_tuple
    print(f"PySpark DataFrame Loaded. Rows: {row_count}")
    print(f"  Load Time (s):       {time_spark_load:.4f}")
    print(f"  Memory Delta (MB):   {mem_spark_load:.4f}\n")


    old_col = "review/text"
    new_col = "review_text"
    if old_col in df_spark.columns:
        df_spark = df_spark.withColumnRenamed(old_col, new_col)

    # Define PySpark UDFs
    @udf(returnType=IntegerType())
    def word_count_udf(text):
        if text is None:
            return 0
        words = re.findall(r'\w+', text)
        return len(words)

    @udf(returnType=IntegerType())
    def sentiment_udf(text):
        if text is None:
            return 0
        return sum(ord(c) for c in text) % 100

    def spark_processing(df_spark_local):
        # Create new columns
        df_spark_processed_local = (
            df_spark_local
            .withColumn("word_count", word_count_udf(F.col("review_text")))
            .withColumn("sentiment_score", sentiment_udf(F.col("review_text")))
        )
        # Trigger actual execution with collect()
        result = df_spark_processed_local.collect()
        return result

    print("PySpark Processing")
    (df_spark_result, time_spark_proc, mem_spark_proc) = measure_memory_and_time(
        spark_processing,
        df_spark
    )
    print(f"PySpark processing completed.")
    print(f"  Processing Time (s): {time_spark_proc:.4f}")
    print(f"  Memory Delta (MB):   {mem_spark_proc:.4f}")
    print(f"Number of rows processed: {len(df_spark_result)}\n")

    spark.stop()


    results = {
        'tool': ['Pandas', 'Dask', 'PySpark'],
        'time_s': [
            time_pandas_proc,
            time_dask_proc,
            time_spark_proc
        ],
        'memory_mb': [
            mem_pandas_proc,
            mem_dask_proc,
            mem_spark_proc
        ]
    }

    results_df = pd.DataFrame(results)
    print("Summary of Processing")
    print(results_df, "\n")

    plt.figure(figsize=(6, 4))
    plt.bar(results_df['tool'], results_df['time_s'])
    plt.title('Execution Time Comparison')
    plt.xlabel('Tool')
    plt.ylabel('Time (s)')
    plt.savefig("Execution Time Comparison.png")

    plt.figure(figsize=(6, 4))
    plt.bar(results_df['tool'], results_df['memory_mb'])
    plt.title('Memory Usage Comparison')
    plt.xlabel('Tool')
    plt.ylabel('Memory (MB)')
    plt.savefig("Memory Usage Comparison.png")
