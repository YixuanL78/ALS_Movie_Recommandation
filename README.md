# Movie Recommendation System Notebook

This repository contains a comprehensive movie recommendation system built with PySpark. The notebook demonstrates a complete end-to-end workflow, starting with data ingestion and exploration, moving through model training and tuning using ALS (Alternating Least Squares), and ending with personalized movie recommendations and similarity analysis.

## Dataset and Data Loading
Data Source: https://grouplens.org/datasets/movielens/latest/
Storage: Data files (movies.csv, ratings.csv, links.csv, tags.csv) are stored on Google Drive and mounted using Colab.
Data Ingestion: Apache Spark is used to read CSV files with headers. The data is loaded into Spark DataFrames for scalable processing.
python

## Data Preprocessing and Exploration
Schema Inspection: The notebook prints the schema and a sample of rows from each DataFrame to understand the data structure.
Handling Null Values: Null value counts are computed for each column to ensure data quality.
Timestamp Conversion: Timestamps in the ratings and tags data are converted into human-readable formats using the from_unixtime function.
Data Cleaning: For instance, the links DataFrame is cleaned using the na.drop() method.
Exploratory Data Analysis (EDA):
Aggregations and counts are performed using Spark SQL.
Unique genre extraction is achieved by exploding the genres field, which helps in understanding the distribution of movie genres.
Spark SQL Analysis
Temporary Views: DataFrames are registered as temporary SQL tables (e.g., movies, ratings, links, and tags) to perform SQL queries directly.
Query Examples:
Counting distinct users and movies.
Extracting unique genres and their counts.
Filtering movies that have ratings.

## ALS Model Training and Tuning
ALS Algorithm: The notebook uses Spark’s ALS algorithm from MLlib to predict movie ratings.
Data Type Conversion: User and movie IDs are cast to integers, and ratings to floats for model compatibility.
Model Tuning:
A parameter grid is defined using ParamGridBuilder to tune hyperparameters like regParam, rank, and maxIter.
Cross-validation is implemented using CrossValidator with 5 folds to find the best model.
Evaluation: The model performance is evaluated using RMSE (Root Mean Squared Error).

## Generating Movie Recommendations
Recommendation Generation: Once the best model is selected, recommendations are generated for all users using the recommendForAllUsers method.
User-Specific Recommendations: A custom function is implemented to extract and display recommended movies for a given user ID.

## Finding Similar Movies
Similarity Metrics: Two similarity metrics are implemented:
Cosine Similarity: To measure the similarity between movie latent factors.
Euclidean Distance: To compute the distance between movie latent features.
Custom Functions: Functions cosine_similarity and euclidean_distance are defined using NumPy, and are applied to the item factors extracted from the best ALS model.
Similarity Search: The notebook defines a function find_similar_movies that, given a movie ID, computes similarity scores and returns the top similar movies.

## Technologies and Libraries Used
Apache Spark & PySpark: For distributed data processing and machine learning.
Google Colab: For interactive notebook development.
MovieLens Dataset: The source of movie data.
Spark MLlib: For building and tuning the recommendation model.
Pandas & NumPy: For handling and manipulating data during similarity computations.
