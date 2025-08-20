# 🎬 Movie Recommendation System

This project implements a **User-Based Collaborative Filtering** recommendation system using the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

---

## 📌 Project Overview
The goal of this project is to recommend movies to users based on the preferences of **similar users**.  
We use **cosine similarity** on a user–item rating matrix and evaluate recommendations with **Precision@K, Recall@K, and F1@K**.

---

## 📂 Dataset
We used the **MovieLens small dataset** which contains:
- 100,836 ratings
- 610 users
- 9,742 movies

Each user rated movies on a scale from **0.5 to 5.0**.

---

## ⚙️ Methodology
1. **Data Preparation**
   - Load ratings and movies data
   - Check for duplicates, missing values
   - Create a `user-item matrix` of users × movies

2. **Model Building**
   - Compute cosine similarity between users
   - For a given user, find **top-K most similar users**
   - Recommend movies that similar users rated highly but the target user hasn’t seen

3. **Evaluation**
   - Train/test split of ratings
   - Define **relevant items** as ratings ≥ 4.0
   - Compute:
     - **Precision@K**
     - **Recall@K**
     - **F1@K**

---

## 📊 Results
| K (Top-N Recommendations) | Precision@K | Recall@K | F1@K  |
|----------------------------|-------------|----------|-------|
| 5                          | 0.0217      | 0.0123   | 0.0155|
| 10                         | 0.0152      | 0.0161   | 0.0156|
| 20                         | 0.0114      | 0.0218   | 0.0117|
