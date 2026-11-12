import os
import sys
import shutil

# Explicitly bind PySpark Python executable to the virtual environment python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Ensure JAVA_HOME is configured to compatible JDK 17
if "JAVA_HOME" not in os.environ or "jdk-17" not in os.environ.get("JAVA_HOME", ""):
    candidates = [
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot",
        r"C:\Program Files\Eclipse Adoptium\jdk-17",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            os.environ["JAVA_HOME"] = cand
            os.environ["PATH"] = os.path.join(cand, "bin") + ";" + os.environ.get("PATH", "")
            break

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Copy data files to data/ if they were placed in root
if os.path.exists("ratings.csv") and not os.path.exists("data/ratings.csv"):
    shutil.copy("ratings.csv", "data/ratings.csv")
if os.path.exists("movies.csv") and not os.path.exists("data/movies.csv"):
    shutil.copy("movies.csv", "data/movies.csv")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 70)
print("  INT315: Large-Scale Movie Lens Analysis and Recommendation System")
print("  Mapping Syllabus: Unit I, Unit III (RDD), Unit IV (Spark SQL), Unit VI (MLlib)")
print("=" * 70)

# ==============================================================================
# SECTION 1: SPARK SESSION INITIALIZATION 
# ==============================================================================
print("\n[Step 1] Initializing Spark Session in Local Cluster Mode...")
spark = (
    SparkSession.builder
    .appName("INT315_MovieLens_Analysis_and_Recommendation")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark_version = spark.version
default_parallelism = spark.sparkContext.defaultParallelism
print(f"-> Spark Version: {spark_version}")
print(f"-> Local Parallelism (Cores utilized): {default_parallelism}")

# ==============================================================================
# SECTION 2: DATA INGESTION & SCHEMA INSPECTION
# ==============================================================================
print("\n[Step 2] Loading MovieLens Small Dataset...")
ratings = spark.read.csv("data/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("data/movies.csv", header=True, inferSchema=True)

print("\n--- Ratings Schema ---")
ratings.printSchema()

print("\n--- Movies Schema ---")
movies.printSchema()

total_ratings_raw = ratings.count()
total_movies_raw = movies.count()
print(f"Raw Ratings Count: {total_ratings_raw:,}")
print(f"Raw Movies Count : {total_movies_raw:,}")

print("\nSample Ratings (First 5):")
ratings.show(5)

print("Sample Movies (First 5):")
movies.show(5, truncate=False)

# ==============================================================================
# SECTION 3: DATA CLEANING & VALIDATION
# ==============================================================================
print("\n[Step 3] Data Cleaning & Duplicate Removal...")
ratings_clean = ratings.dropna().dropDuplicates()
movies_clean = movies.dropna().dropDuplicates()

cleaned_ratings_count = ratings_clean.count()
cleaned_movies_count = movies_clean.count()
print(f"Cleaned Ratings Count: {cleaned_ratings_count:,} (Nulls/Duplicates Removed: {total_ratings_raw - cleaned_ratings_count})")
print(f"Cleaned Movies Count : {cleaned_movies_count:,} (Nulls/Duplicates Removed: {total_movies_raw - cleaned_movies_count})")

# Cache cleaned DataFrames for efficient downstream operations
ratings_clean.cache()
movies_clean.cache()

# ==============================================================================
# SECTION 4: UNIT III SYLLABUS - RDD OPERATIONS & METHODS
# ==============================================================================
print("\n[Step 4 - Unit III] Demonstrating Spark RDD Creation, Transformations & Actions...")
# 4.1 Convert DataFrame to RDD
ratings_rdd = ratings_clean.rdd
print("-> Successfully converted DataFrame to RDD")

# 4.2 RDD Transformation: filter high ratings (>= 4.5)
high_ratings_rdd = ratings_rdd.filter(lambda row: row.rating >= 4.5)

# 4.3 RDD Action: count
high_ratings_count = high_ratings_rdd.count()
print(f"-> RDD Action [count]: Number of high ratings (>= 4.5 stars): {high_ratings_count:,}")

# 4.4 RDD Transformation [map]: Extract (userId, rating) pairs
user_rating_pairs_rdd = ratings_rdd.map(lambda row: (row.userId, row.rating))

# 4.5 RDD Action [take]: View first 3 elements
sample_rdd_elements = user_rating_pairs_rdd.take(3)
print(f"-> RDD Action [take(3)]: Sample (userId, rating) tuples: {sample_rdd_elements}")

# ==============================================================================
# SECTION 5: DATA ANALYSIS & AGGREGATIONS
# ==============================================================================
print("\n[Step 5] Exploratory Data Analysis with PySpark DataFrames...")

# 5.1 Rating Distribution
print("\n=== Rating Distribution ===")
dist = ratings_clean.groupBy("rating").count().orderBy("rating")
dist.show()
dist_pd = dist.toPandas()
dist_pd.to_csv("outputs/rating_distribution.csv", index=False)

# 5.2 Most Popular Movies (Highest Rating Counts)
print("\n=== Top 10 Most Rated Movies ===")
most_rated = (
    ratings_clean.groupBy("movieId")
    .agg(
        F.count("rating").alias("num_ratings"),
        F.round(F.avg("rating"), 2).alias("avg_rating")
    )
    .join(movies_clean, "movieId")
    .orderBy(F.desc("num_ratings"))
)
top_10_most_rated = most_rated.select("movieId", "title", "num_ratings", "avg_rating")
top_10_most_rated.show(10, truncate=False)
top_10_most_rated_pd = top_10_most_rated.limit(10).toPandas()
top_10_most_rated_pd.to_csv("outputs/top_10_most_rated_movies.csv", index=False)

# 5.3 Highest Rated Movies with statistically significant threshold (>= 50 ratings)
print("\n=== Top 10 Highest Rated Movies (Min 50 Ratings) ===")
top_10_highest_rated = (
    most_rated.filter(F.col("num_ratings") >= 50)
    .orderBy(F.desc("avg_rating"), F.desc("num_ratings"))
    .select("movieId", "title", "num_ratings", "avg_rating")
)
top_10_highest_rated.show(10, truncate=False)
top_10_highest_rated_pd = top_10_highest_rated.limit(10).toPandas()
top_10_highest_rated_pd.to_csv("outputs/top_10_highest_rated_movies.csv", index=False)

# 5.4 Genre Analysis (explode genres array)
print("\n=== Movie Genre Statistics ===")
genres_df = movies_clean.withColumn("genre", F.explode(F.split("genres", r"\|")))
genre_stats = (
    ratings_clean.join(genres_df, "movieId")
    .groupBy("genre")
    .agg(
        F.count("rating").alias("num_ratings"),
        F.round(F.avg("rating"), 2).alias("avg_rating")
    )
    .orderBy(F.desc("num_ratings"))
)
genre_stats.show(25, truncate=False)
genre_stats_pd = genre_stats.toPandas()
genre_stats_pd.to_csv("outputs/genre_statistics.csv", index=False)

# 5.5 Most Active Users
print("\n=== Top 10 Most Active Users ===")
active_users = (
    ratings_clean.groupBy("userId")
    .agg(
        F.count("rating").alias("rating_count"),
        F.round(F.avg("rating"), 2).alias("avg_user_rating")
    )
    .orderBy(F.desc("rating_count"))
)
active_users.show(10)
active_users_pd = active_users.limit(10).toPandas()
active_users_pd.to_csv("outputs/top_10_active_users.csv", index=False)

# ==============================================================================
# SECTION 6: UNIT IV SYLLABUS - SPARK SQL QUERIES & SQL WILDCARDS
# ==============================================================================
print("\n[Step 6 - Unit IV] Executing Spark SQL Queries...")
ratings_clean.createOrReplaceTempView("ratings")
movies_clean.createOrReplaceTempView("movies")

# Query 1: Temporal Trend Analysis (Aggregation, GroupBy, OrderBy)
print("\n--- Spark SQL Query 1: Rating Counts and Average Rating by Year ---")
yearly_query = """
    SELECT 
        year(from_unixtime(timestamp)) AS release_year,
        COUNT(*) AS total_ratings,
        ROUND(AVG(rating), 2) AS avg_rating
    FROM ratings
    GROUP BY year(from_unixtime(timestamp))
    ORDER BY release_year
"""
yearly_stats = spark.sql(yearly_query)
yearly_stats.show(30)
yearly_stats_pd = yearly_stats.toPandas()
yearly_stats_pd.to_csv("outputs/yearly_rating_trend.csv", index=False)

# Query 2: SQL Wildcards (LIKE '%Star Wars%') demonstrating Unit IV syllabus
print("\n--- Spark SQL Query 2: SQL Wildcard Analysis ('Star Wars%' Franchise) ---")
franchise_query = """
    SELECT 
        m.movieId,
        m.title,
        m.genres,
        COUNT(r.rating) AS total_reviews,
        ROUND(AVG(r.rating), 2) AS average_rating
    FROM movies m
    JOIN ratings r ON m.movieId = r.movieId
    WHERE m.title LIKE 'Star Wars%'
    GROUP BY m.movieId, m.title, m.genres
    ORDER BY total_reviews DESC
"""
franchise_stats = spark.sql(franchise_query)
franchise_stats.show(10, truncate=False)
franchise_stats_pd = franchise_stats.toPandas()
franchise_stats_pd.to_csv("outputs/star_wars_franchise_sql.csv", index=False)

# ==============================================================================
# SECTION 7: UNIT VI SYLLABUS - SPARK MLlib COLLABORATIVE FILTERING (ALS)
# ==============================================================================
print("\n[Step 7 - Unit VI] Building ALS Recommendation Model with PySpark MLlib...")

# 7.1 Cast columns to types required by Spark MLlib ALS
ratings_cast = ratings_clean.select(
    F.col("userId").cast("int"),
    F.col("movieId").cast("int"),
    F.col("rating").cast("float")
)

# 7.2 Train/Test Split (80% Training, 20% Testing)
train, test = ratings_cast.randomSplit([0.8, 0.2], seed=42)
train_count = train.count()
test_count = test.count()
print(f"-> Training Set Size (80%): {train_count:,} rows")
print(f"-> Testing Set Size  (20%): {test_count:,} rows")

# 7.3 Model Hyperparameters and Fitting
print("\n-> Fitting ALS Model (rank=10, maxIter=10, regParam=0.1, coldStartStrategy='drop')...")
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)
model = als.fit(train)

# 7.4 Predictions on Test Set
predictions = model.transform(test)
print("\n=== Model Predictions on Test Dataset ===")
predictions.show(10)

# Save sample predictions joined with movie titles
sample_preds = (
    predictions.join(movies_clean, "movieId")
    .select("userId", "movieId", "title", "rating", F.round("prediction", 2).alias("predicted_rating"))
    .limit(20)
)
sample_preds_pd = sample_preds.toPandas()
sample_preds_pd.to_csv("outputs/sample_predictions.csv", index=False)

# 7.5 Model Evaluation Metrics: RMSE & MAE (Unit VI Evaluation Matrix)
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

rmse_val = evaluator_rmse.evaluate(predictions)
mae_val = evaluator_mae.evaluate(predictions)

print("\n" + "=" * 50)
print(f"  MODEL EVALUATION METRICS (Unit VI):")
print(f"  Root Mean Squared Error (RMSE) : {rmse_val:.4f}")
print(f"  Mean Absolute Error (MAE)      : {mae_val:.4f}")
print("=" * 50)

# 7.6 Top 5 Movie Recommendations for Sample Users
print("\n=== Generating Top 5 Movie Recommendations for Sample Users (Users 1-5) ===")
sample_users = [1, 2, 3, 4, 5]
sample_users_df = spark.createDataFrame([(uid,) for uid in sample_users], ["userId"])
user_recs_raw = model.recommendForUserSubset(sample_users_df, 5)

user_recommendations = (
    user_recs_raw
    .select("userId", F.explode("recommendations").alias("rec"))
    .select(
        "userId",
        F.col("rec.movieId").alias("movieId"),
        F.round("rec.rating", 2).alias("predicted_rating")
    )
    .join(movies_clean, "movieId")
    .select("userId", "movieId", "title", "genres", "predicted_rating")
    .orderBy("userId", F.desc("predicted_rating"))
)
user_recommendations.show(25, truncate=False)
user_recs_pd = user_recommendations.toPandas()
user_recs_pd.to_csv("outputs/recommendations_for_sample_users.csv", index=False)

# ==============================================================================
# SECTION 8: VISUALIZATIONS WITH MATPLOTLIB
# ==============================================================================
print("\n[Step 8] Generating and Saving Publication-Quality Visualizations...")

# Chart 1: Rating Distribution
plt.figure(figsize=(8, 4.5))
plt.bar(dist_pd["rating"].astype(str), dist_pd["count"], color="#1f77b4", edgecolor="black", alpha=0.85)
plt.title("Distribution of Movie Ratings (MovieLens Small)", fontsize=13, fontweight="bold")
plt.xlabel("Rating (Stars)", fontsize=11)
plt.ylabel("Number of Ratings", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png", dpi=300)
plt.savefig("rating_distribution.png", dpi=300)
plt.close()
print("-> Saved: outputs/rating_distribution.png")

# Chart 2: Top 10 Genres by Ratings Count
plt.figure(figsize=(10, 5))
genre_top10 = genre_stats_pd.head(10)
plt.barh(genre_top10["genre"], genre_top10["num_ratings"], color="#ff7f0e", edgecolor="black", alpha=0.85)
plt.title("Top 10 Movie Genres by Frequency of Ratings", fontsize=13, fontweight="bold")
plt.xlabel("Total Ratings Count", fontsize=11)
plt.ylabel("Genre", fontsize=11)
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/genre_ratings.png", dpi=300)
plt.savefig("genre_ratings.png", dpi=300)
plt.close()
print("-> Saved: outputs/genre_ratings.png")

# Chart 3: Yearly Ratings Trend (Spark SQL Output Visualization)
plt.figure(figsize=(10, 4.5))
plt.plot(yearly_stats_pd["release_year"], yearly_stats_pd["total_ratings"], marker="o", color="#2ca02c", linewidth=2)
plt.title("Yearly Movie Rating Activity (Spark SQL Trend Analysis)", fontsize=13, fontweight="bold")
plt.xlabel("Year", fontsize=11)
plt.ylabel("Ratings Submitted", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/yearly_rating_trend.png", dpi=300)
plt.savefig("yearly_rating_trend.png", dpi=300)
plt.close()
print("-> Saved: outputs/yearly_rating_trend.png")

# ==============================================================================
# SECTION 9: SUMMARY REPORT GENERATION
# ==============================================================================
most_frequent_rating_row = dist_pd.sort_values(by="count", ascending=False).iloc[0]
top_movie_row = top_10_most_rated_pd.iloc[0]
highest_rated_row = top_10_highest_rated_pd.iloc[0]
top_genre_row = genre_stats_pd.iloc[0]

summary_content = f"""================================================================================
INT315 PROJECT EXECUTION RUN SUMMARY: MOVIELENS ANALYSIS & RECOMMENDATION
================================================================================
Syllabus Alignment:
  - Unit I  : Introduction to Apache Spark (Standalone / Local Cluster Mode)
  - Unit III: Spark RDDs (Creation, Transformations, Actions, RDD-to-DataFrame)
  - Unit IV : Spark SQL (Temporary Views, Aggregations, Joins, SQL Wildcards)
  - Unit VI : Spark MLlib (Collaborative Filtering with ALS, Evaluation with RMSE/MAE)

Execution Environment:
  - Spark Version             : {spark_version}
  - Parallelism (Worker Cores): {default_parallelism}
  - Master Connection         : local[*]

Data Metrics:
  - Raw Ratings Count         : {total_ratings_raw:,}
  - Raw Movies Count          : {total_movies_raw:,}
  - Cleaned Ratings Count     : {cleaned_ratings_count:,}
  - Cleaned Movies Count      : {cleaned_movies_count:,}
  - Training Set (80%)        : {train_count:,} rows
  - Testing Set (20%)         : {test_count:,} rows

Key Analytical Findings:
  - Most Frequent Rating      : {most_frequent_rating_row['rating']} stars ({most_frequent_rating_row['count']:,} ratings)
  - Most Rated Movie          : {top_movie_row['title']} ({top_movie_row['num_ratings']:,} ratings, avg {top_movie_row['avg_rating']})
  - Highest Rated Movie (>=50): {highest_rated_row['title']} (avg {highest_rated_row['avg_rating']})
  - Top Rated Genre           : {top_genre_row['genre']} ({top_genre_row['num_ratings']:,} ratings, avg {top_genre_row['avg_rating']})
  - High Ratings (>=4.5) [RDD]: {high_ratings_count:,}

Machine Learning Evaluation (Unit VI):
  - Model Algorithm           : Alternating Least Squares (ALS) Collaborative Filtering
  - Hyperparameters           : rank=10, maxIter=10, regParam=0.1, nonnegative=True
  - Root Mean Squared Error   : {rmse_val:.4f}
  - Mean Absolute Error (MAE) : {mae_val:.4f}

Artifacts Generated in outputs/:
  1. outputs/run_summary.txt
  2. outputs/rating_distribution.csv
  3. outputs/top_10_most_rated_movies.csv
  4. outputs/top_10_highest_rated_movies.csv
  5. outputs/genre_statistics.csv
  6. outputs/top_10_active_users.csv
  7. outputs/yearly_rating_trend.csv
  8. outputs/star_wars_franchise_sql.csv
  9. outputs/sample_predictions.csv
  10. outputs/recommendations_for_sample_users.csv
  11. outputs/rating_distribution.png
  12. outputs/genre_ratings.png
  13. outputs/yearly_rating_trend.png
================================================================================
"""

with open("outputs/run_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_content)

print("\n" + summary_content)

spark.stop()
print("PySpark execution completed successfully!")
