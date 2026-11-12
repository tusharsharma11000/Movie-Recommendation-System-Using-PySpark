
# MovieLens Recommendation Engine

A distributed movie analytics and recommendation engine built using **Python, Apache Spark, PySpark, Spark SQL, and Spark MLlib**. The project analyzes movie ratings, identifies user preferences, and generates personalized movie recommendations using collaborative filtering.

## Overview

This project demonstrates how distributed data processing and machine learning can be combined to build a movie recommendation system.

It processes the MovieLens dataset using Apache Spark, performs analytical queries using Spark SQL, and applies the Alternating Least Squares (ALS) algorithm to generate personalized recommendations.

## Key Features

- **Distributed Data Processing:** Process movie and rating datasets using PySpark.
- **Data Analysis:** Analyze movie popularity, user activity, rating distributions, and genre preferences.
- **Spark SQL:** Perform analytical queries, aggregations, and temporal analysis.
- **Collaborative Filtering:** Generate personalized recommendations using the ALS algorithm.
- **Model Evaluation:** Evaluate recommendation accuracy using RMSE and MAE.
- **Automated Reporting:** Generate analytical reports, CSV outputs, and visualizations.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Big Data Processing | Apache Spark, PySpark |
| Query Engine | Spark SQL |
| Machine Learning | Spark MLlib, ALS |
| Data Processing | RDDs, DataFrames |
| Visualization | Matplotlib |
| Dataset | GroupLens MovieLens |

## System Architecture

```text
          MovieLens Dataset
           Movies + Ratings
                  |
                  v
          Data Processing
             PySpark
                  |
                  v
          Analytical Engine
             Spark SQL
                  |
          +-------+-------+
          |               |
          v               v
     Movie Analytics   User Analytics
          |               |
          +-------+-------+
                  |
                  v
        Recommendation Engine
             MLlib ALS
                  |
                  v
           Model Evaluation
             RMSE / MAE
                  |
                  v
        Personalized Movie
          Recommendations
                  |
                  v
          Reports & Outputs
```

## Dataset

The project uses the official **MovieLens Dataset** provided by GroupLens Research.

Dataset Source: https://grouplens.org/datasets/movielens/

| Dataset | Records |
|---|---:|
| Movies | 9,742 |
| Ratings | 100,836 |

## Implementation

### 1. Distributed Data Processing
- Loaded and processed movie and rating datasets using PySpark.
- Applied RDD transformations and actions.
- Used DataFrame operations for structured data processing.
- Configured Spark to utilize available CPU cores in local execution mode.

### 2. Data Analytics with Spark SQL
- Analyzed movie rating distributions.
- Identified the most popular and highest-rated movies.
- Performed genre-wise rating analysis.
- Analyzed yearly rating trends.
- Used SQL queries and temporary views for data exploration.

### 3. Movie Recommendation Engine
- Implemented the Alternating Least Squares (ALS) algorithm.
- Applied collaborative filtering to user-movie rating data.
- Divided the dataset into 80% training and 20% testing data.
- Generated personalized top-5 movie recommendations for sample users.

### 4. Model Evaluation

The recommendation model was evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| RMSE | 0.8672 |
| MAE | 0.6700 |
| Training Records | 80,877 |
| Testing Records | 19,959 |

## Key Results

- Processed 100,836 movie ratings and 9,742 movie records.
- Identified Forrest Gump (1994) as the most frequently rated movie with 329 ratings.
- Identified The Shawshank Redemption (1994) as the highest-rated movie among titles with at least 50 ratings.
- Analyzed genre preferences and rating patterns.
- Generated personalized movie recommendations using ALS.
- Evaluated recommendation performance using RMSE and MAE.

## Project Structure

```text
MovieLens-Recommendation-Engine/
│
├── data/
│   ├── ratings.csv
│   └── movies.csv
│
├── outputs/
│   ├── run_summary.txt
│   ├── rating_distribution.csv
│   ├── top_10_most_rated_movies.csv
│   ├── top_10_highest_rated_movies.csv
│   ├── genre_statistics.csv
│   ├── top_10_active_users.csv
│   ├── yearly_rating_trend.csv
│   ├── star_wars_franchise_sql.csv
│   ├── sample_predictions.csv
│   ├── recommendations_for_sample_users.csv
│   ├── rating_distribution.png
│   ├── genre_ratings.png
│   └── yearly_rating_trend.png
│
├── movie_analysis.py
├── generate_report_docx.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Java JDK 17
- Apache Spark 3.5.3

### Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MovieLens-Recommendation-Engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Project

Execute the main analysis pipeline:

```bash
python movie_analysis.py
```

Generate the Word report:

```bash
python generate_report_docx.py
```

The generated analytical results, recommendation outputs, and visualizations are saved in the `outputs/` directory.

## Future Improvements

- Develop a REST API for serving movie recommendations.
- Build an interactive web interface for personalized recommendations.
- Integrate real-time rating updates.
- Explore cloud-based distributed deployment.
- Improve recommendation performance through model tuning.

## Author

**Tushar Sharma**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/tusharsharma11000
