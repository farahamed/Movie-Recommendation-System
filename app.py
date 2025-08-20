import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split




ratings=pd.read_csv('C:/Users/DELL/Desktop/ML learning internship/Movie Recommendation System Description Task 3/Dataset/ml-latest-small/ratings.csv')
movies=pd.read_csv('C:/Users/DELL/Desktop/ML learning internship/Movie Recommendation System Description Task 3/Dataset/ml-latest-small/movies.csv')
print("ratings:\n")
print(ratings.head())
print(ratings.info())
print("Statistical Information:\n",ratings.describe())
print("Duplicated Rows:\n",ratings.duplicated().sum())
print("is null rows:\n", ratings.isnull().sum())
print("Ratings:\n",ratings["rating"].value_counts())
print("how many users?",ratings["userId"].nunique())
print("shape:\n",ratings.shape)

print("Movies:\n")
print(movies.head())
print(movies.info())
print("Statistical Information:\n",movies.describe())
print("Duplicated Rows:\n",movies.duplicated().sum())
print("is null rows:\n", movies.isnull().sum())
print("how many movies?",ratings["movieId"].nunique())
print("shape:\n",movies.shape)

user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
similarity = cosine_similarity(user_item_matrix)

similarity_dataframe = pd.DataFrame(
    similarity, 
    index=user_item_matrix.index,   
    columns=user_item_matrix.index  
)


def recommended_movie(userId,k):
    similar_scores=similarity_dataframe.loc[userId]
    sorted_scores=similar_scores.sort_values(ascending=False)
    top_users = sorted_scores.index[1:k+1]
    predicted_ratings=user_item_matrix.loc[top_users].mean(axis=0)
    recommended_movies = pd.DataFrame({
    'movieId': predicted_ratings.index,
    'predicted_rating': predicted_ratings.values
    })
    recommended_movies = recommended_movies.merge(movies, on='movieId')
    watched = user_item_matrix.loc[userId]
    unseen = recommended_movies[~recommended_movies['movieId'].isin(watched[watched > 0].index)]
    return unseen





train_ratings, test_ratings = train_test_split(ratings,test_size=0.2, random_state=42
)

user_item_matrix = train_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)



relevance_threshold = 4.0


user_relevant = (
    test_ratings[test_ratings['rating'] >= relevance_threshold]
    .groupby('userId')['movieId']
    .apply(list)
)

print(user_relevant.head())
K = 20  
all_precision = []
all_recall = []
all_f1=[]

for userId in user_relevant.index:   
    
    
    relevant_items = user_relevant[userId]
    
   
    recommended_items = recommended_movie(userId, K)['movieId'].head(K).tolist()
    
   
    hits = len(set(recommended_items) & set(relevant_items))
    
    
    precision_at_k = hits / K
    recall_at_k = hits / len(relevant_items) if len(relevant_items) > 0 else 0
    
    
    all_precision.append(precision_at_k)
    all_recall.append(recall_at_k)

    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0
    all_f1.append(f1_at_k)


final_precision = np.mean(all_precision)
final_recall = np.mean(all_recall)
final_f1 = np.mean(all_f1)
print(f"Final Precision@{K}: {final_precision:.4f}")
print(f"Final Recall@{K}: {final_recall:.4f}")
print(f"Final F1@{K}: {final_f1:.4f}")



