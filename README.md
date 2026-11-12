# INT315: Large-Scale Movie Ratings Analysis and Recommendation using PySpark

An end-to-end Big Data project built using **Apache Spark (PySpark)**, **Spark SQL**, and **Spark MLlib** on the official **GroupLens MovieLens** dataset. This project directly addresses Course Outcomes **CO4** and **CO5** of the **INT315: Distributed and Cluster Computing** syllabus.

---

## 📋 Course Syllabus Alignment

| INT315 Syllabus Unit | Topic Covered in Project | Implementation Detail |
| :--- | :--- | :--- |
| **Unit I: Introduction to Spark** | In-memory distributed architecture, Standalone local cluster mode | `local[*]` SparkSession utilizing all 12 CPU cores with optimized memory & shuffle partitions |
| **Unit III: RDDs for Big Data Applications** | RDD creation, transformations, actions | DataFrame-to-RDD conversion, `filter` (ratings >= 4.5), `map` pair RDDs, `count`, and `take` actions |
| **Unit IV: Spark SQL** | Relational queries, TempViews, aggregations, SQL wildcards | Registered TempViews, temporal release year grouping, SQL wildcards (`WHERE title LIKE 'Star Wars%'`) |
| **Unit VI: Spark MLlib & PySpark** | Collaborative Filtering (ALS), Evaluation matrix | Matrix Factorization with ALS (rank=10, maxIter=10), evaluated via **RMSE (0.8672)** and **MAE (0.6700)** |

---

## 📁 Project Directory Layout

```text
INT315_Project/
├── data/
│   ├── ratings.csv                           # 100,836 movie ratings
│   └── movies.csv                            # 9,742 movie titles and genres
├── outputs/
│   ├── run_summary.txt                       # Complete terminal summary & execution log
│   ├── rating_distribution.csv               # Frequency count for all star ratings (0.5 to 5.0)
│   ├── top_10_most_rated_movies.csv          # Top 10 movies by review volume
│   ├── top_10_highest_rated_movies.csv       # Highest rated movies (filtered for >= 50 reviews)
│   ├── genre_statistics.csv                  # Ratings count and average stars per genre
│   ├── top_10_active_users.csv               # Top 10 users by submitted review volume
│   ├── yearly_rating_trend.csv               # Spark SQL temporal ratings activity by year
│   ├── star_wars_franchise_sql.csv           # Spark SQL wildcard analysis for Star Wars franchise
│   ├── sample_predictions.csv                # Test set actual vs. ALS predicted ratings
│   ├── recommendations_for_sample_users.csv  # Top 5 personalized movie recs for sample users
│   ├── rating_distribution.png               # High-res chart of rating distribution
│   ├── genre_ratings.png                     # High-res chart of top genres
│   └── yearly_rating_trend.png               # High-res chart of yearly review volume
├── movie_analysis.py                         # Complete PySpark pipeline script
├── generate_report_docx.py                   # Automated Word report builder (.docx)
├── requirements.txt                          # Pinned, tested dependencies
├── INT315_Project_Report.docx                # Complete 12-section project report document
└── README.md                                 # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
- **Python**: 3.10 or 3.11 (tested with Python 3.11.4)
- **Java**: Java JDK 17 (Eclipse Adoptium Temurin 17 installed)
- **Virtual Environment**: `.venv`

### 2. Execution Commands
Run the pipeline from your VS Code PowerShell terminal:
```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run the PySpark analysis pipeline
python movie_analysis.py

# Re-generate the Word Report (.docx) if needed
python generate_report_docx.py
```

---

## 📊 Key Results & Findings

- **Spark Cluster Environment**: Apache Spark 3.5.3 running on `local[*]` with 12 parallel CPU cores.
- **Data Volume**: 100,836 user ratings and 9,742 movies cleaned and processed in seconds.
- **Most Frequent Rating**: **4.0 Stars** (26,818 ratings, 26.6% of the dataset).
- **Most Popular Movie**: *Forrest Gump (1994)* with **329 ratings** (avg: 4.16).
- **Highest Rated Movie (≥50 ratings)**: *The Shawshank Redemption (1994)* with **4.43 / 5.0 stars** (317 ratings).
- **Top Genre**: **Drama** with **41,928 ratings** (avg: 3.66), followed by **Comedy** (39,053 ratings).
- **Highest Rated Genre**: **Film-Noir** (avg: 3.92 stars) and **War** (avg: 3.81 stars).
- **Machine Learning Evaluation (Unit VI)**:
  - **Algorithm**: Alternating Least Squares (ALS) Collaborative Filtering
  - **Train/Test Split**: 80% (80,877 rows) / 20% (19,959 rows)
  - **Root Mean Squared Error (RMSE)**: **0.8672** (Well within standard 0.85–1.00 benchmark)
  - **Mean Absolute Error (MAE)**: **0.6700**

---

## 📑 Report Document

The official submission document is saved as:
👉 **`INT315_Project_Report.docx`**

It contains all 12 comprehensive sections, including:
1. Executive Summary
2. Problem Statement
3. Objectives & Course Outcomes (CO4, CO5)
4. Dataset Description
5. Cluster Hardware & Software Environment
6. Implementation Methodology & Syllabus Mapping (Units I, III, IV, VI)
7. Experimental Results & Discussion
8. MLlib Model Evaluation & Personalization Analysis
9. Screenshot & Submission Checklist
10. Complete Code Listing
11. Conclusion & Future Scope (Kafka Streaming, GraphX)
12. References (Textbooks & Research)
