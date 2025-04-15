import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv('final_steam_user_data.csv')

# Basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique users: {df['steam_id'].nunique()}")
print(f"Number of unique games: {df['app_id'].nunique()}")


# Clean and preprocess the data
def clean_owners(owners_str):
    """Convert owners range to the average number."""
    if pd.isnull(owners_str) or owners_str == "":
        return 0
    # Extract numbers from strings like "5,000,000 .. 10,000,000"
    numbers = re.findall(r'[\d,]+', owners_str)
    if len(numbers) == 2:
        # Convert to integers, removing commas
        low = int(numbers[0].replace(',', ''))
        high = int(numbers[1].replace(',', ''))
        return (low + high) / 2
    return 0


def parse_list_field(field_str):
    """Parse comma-separated fields into lists."""
    if pd.isnull(field_str) or field_str == "":
        return []
    return [item.strip() for item in field_str.split(',')]


# Clean the dataset
print("Cleaning and preprocessing the data...")
df['owners_count'] = df['owners'].apply(clean_owners)
df['tags_list'] = df['tags'].apply(parse_list_field)
df['genres_list'] = df['genre'].apply(parse_list_field)
df['languages_list'] = df['languages'].apply(parse_list_field)

# Create a user-game interaction matrix based on playtime
print("Creating user-game interaction matrix...")
# We'll use average_forever as our interaction strength
user_game_matrix = df.pivot_table(
    index='steam_id',
    columns='app_id',
    values='average_forever',
    fill_value=0
)

# Convert sparse matrix for memory efficiency
user_game_sparse = csr_matrix(user_game_matrix.values)

print(f"User-game matrix shape: {user_game_matrix.shape}")

# --- CONTENT-BASED FILTERING ---
print("\nBuilding content-based filtering model...")

# Create game features from tags, genres, developer, publisher
# First, let's create a dataframe with unique games
games_df = df.drop_duplicates('app_id')[
    ['app_id', 'name', 'developer', 'publisher', 'tags_list', 'genres_list', 'price']]
games_df = games_df.dropna(subset=['name'])
# One-hot encode tags
mlb = MultiLabelBinarizer()
tags_encoded = mlb.fit_transform(games_df['tags_list'])
tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

# One-hot encode genres
genres_encoded = mlb.fit_transform(games_df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# One-hot encode developer and publisher
dev_encoded = pd.get_dummies(games_df['developer'], prefix='dev')
pub_encoded = pd.get_dummies(games_df['publisher'], prefix='pub')

# Combine all features
game_features = pd.concat([tags_df, genres_df, dev_encoded, pub_encoded], axis=1)

# Ensure all DataFrames have the same indices
# Reset index to make sure we maintain the correct ordering
games_df = games_df.reset_index(drop=True)
game_features = game_features.reset_index(drop=True)

# Fill NaN values with 0 in game_features
game_features = game_features.fillna(0)

# Normalize price
scaler = MinMaxScaler()
games_df['price_scaled'] = scaler.fit_transform(games_df[['price']].fillna(0))

# Add price to game features, ensuring matching indices
game_features = game_features.loc[games_df.index]  # Match indices
game_features['price'] = games_df['price_scaled']

# Final check to ensure no NaN values
game_features = game_features.fillna(0)
print(f"NaN values in game_features: {game_features.isna().sum().sum()}")

print(f"Game features shape: {game_features.shape}")

# Calculate content-based similarity
print("Computing content similarity matrix (this may take some time)...")

# Instead of using sparse matrix which is causing data type issues,
# we'll compute similarity directly but with memory optimization
# First, convert to numeric type explicitly
game_features = game_features.astype(float)


# If the feature matrix is too large, we can use batch processing
# to compute cosine similarity
def compute_cosine_similarity_in_batches(matrix, batch_size=1000):
    n_samples = matrix.shape[0]
    sim_matrix = np.zeros((n_samples, n_samples))

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = matrix[i:batch_end]

        # Compute similarities for this batch
        batch_sim = cosine_similarity(batch, matrix)
        sim_matrix[i:batch_end] = batch_sim

        # Report progress
        if i % 5000 == 0:
            print(f"  Processed {i}/{n_samples} rows")

    return sim_matrix


content_sim = compute_cosine_similarity_in_batches(game_features.values)
print("Content-based similarity matrix computed")


# Function to get content-based recommendations
def get_content_recommendations(game_id, top_n=10):
    """Get content-based recommendations based on game similarity."""
    if game_id not in games_df['app_id'].values:
        return pd.DataFrame()

    # Find the index of the game in our dataframe
    idx = games_df[games_df['app_id'] == game_id].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Exclude the game itself

    # Get game indices
    game_indices = [i[0] for i in sim_scores]

    # Return the recommended games with similarity scores
    recommendations = games_df.iloc[game_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return recommendations[['app_id', 'name', 'similarity_score']]


# --- COLLABORATIVE FILTERING ---
print("\nBuilding collaborative filtering model...")

# Split data for training and testing
# First convert to csr_matrix to save memory
user_game_sparse = csr_matrix(user_game_matrix.values)

# Use a smaller test size to prevent memory issues
print("Splitting data into training and test sets...")
train_ratio = 0.8
test_ratio = 0.2

# Get dimensions
n_users, n_items = user_game_sparse.shape

# Create masks for train and test
from scipy import sparse
import random

# For each user, create train and test sets
train_matrix = sparse.lil_matrix(user_game_sparse.shape)
test_matrix = sparse.lil_matrix(user_game_sparse.shape)

for u in range(n_users):
    # Get indices and data of items this user has interacted with
    row = user_game_sparse[u]
    indices = row.nonzero()[1]
    data = row.data

    # Skip users with no interactions
    if len(indices) == 0:
        continue

    # Number of items to move to test
    n_test = max(1, int(len(indices) * test_ratio))

    # Randomly select items to move to test
    test_idx = random.sample(range(len(indices)), n_test)

    # Add to train and test matrices
    for i, idx in enumerate(indices):
        if i in test_idx:
            test_matrix[u, idx] = data[i]
        else:
            train_matrix[u, idx] = data[i]

# Convert to CSR format for faster operations
train_matrix = train_matrix.tocsr()
test_matrix = test_matrix.tocsr()

print(
    f"Train matrix: {train_matrix.shape}, density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
print(
    f"Test matrix: {test_matrix.shape}, density: {test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1]):.6f}")

# Train a KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model.fit(train_matrix)

print("KNN model trained for collaborative filtering")


# Function to get collaborative filtering recommendations
def get_collaborative_recommendations(user_id, top_n=10):
    """Get collaborative filtering recommendations for a user."""
    if user_id not in user_game_matrix.index:
        return pd.DataFrame()

    # Find the index of the user
    user_idx = user_game_matrix.index.get_loc(user_id)

    # Get the user's game interactions
    user_vector = train_matrix[user_idx:user_idx + 1]

    # Find similar users
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=10)

    # Get the games played by similar users
    similar_users_indices = indices.flatten()

    # Create a dictionary to store game scores
    game_scores = {}

    # Iterate through similar users
    for idx in similar_users_indices:
        if idx == user_idx:  # Skip the user itself
            continue

        # Get the user's vector
        user_vec = train_matrix[idx].toarray().flatten()

        # Find games the user has played
        played_games = np.where(user_vec > 0)[0]

        # Score the games
        for game_idx in played_games:
            game_id = user_game_matrix.columns[game_idx]

            # Skip games the target user has already played
            if user_game_matrix.iloc[user_idx, game_idx] > 0:
                continue

            # Calculate a score based on similarity and playtime
            if game_id not in game_scores:
                game_scores[game_id] = 0

            # Add to the score (inversely weighted by distance)
            distance_weight = 1 / (1 + distances[0][list(similar_users_indices).index(idx)])
            game_scores[game_id] += user_vec[game_idx] * distance_weight

    # Sort games by score
    sorted_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Create a dataframe with the recommended games
    recommendations = pd.DataFrame(sorted_games, columns=['app_id', 'cf_score'])

    # Add game details
    game_details = games_df[games_df['app_id'].isin(recommendations['app_id'])]
    recommendations = recommendations.merge(game_details[['app_id', 'name']], on='app_id')

    return recommendations[['app_id', 'name', 'cf_score']]


# --- HYBRID RECOMMENDER ---
print("\nBuilding hybrid recommender system...")


def get_hybrid_recommendations(user_id, content_weight=0.4, top_n=10):
    """Get hybrid recommendations combining collaborative and content-based filtering."""
    # Get user's play history
    if user_id not in user_game_matrix.index:
        return pd.DataFrame()

    user_history = user_game_matrix.loc[user_id]
    user_history = user_history[user_history > 0]

    if len(user_history) == 0:
        return pd.DataFrame()  # No play history

    # Get the most played game
    most_played_game = user_history.idxmax()

    # Get content-based recommendations based on the most played game
    content_recs = get_content_recommendations(most_played_game, top_n=top_n * 2)

    # Get collaborative filtering recommendations
    collab_recs = get_collaborative_recommendations(user_id, top_n=top_n * 2)

    # If we couldn't get recommendations from one method, return the other
    if content_recs.empty:
        return collab_recs.head(top_n)
    if collab_recs.empty:
        return content_recs.head(top_n)

    # Normalize scores for each method
    content_recs['norm_score'] = content_recs['similarity_score'] / content_recs['similarity_score'].max()
    collab_recs['norm_score'] = collab_recs['cf_score'] / collab_recs['cf_score'].max()

    # Combine the recommendations
    # First, get all unique game IDs
    all_games = set(content_recs['app_id']).union(set(collab_recs['app_id']))

    # Create a dictionary to store hybrid scores
    hybrid_scores = {}

    for game_id in all_games:
        # Initialize scores
        content_score = 0
        collab_score = 0

        # Get content-based score if available
        content_game = content_recs[content_recs['app_id'] == game_id]
        if not content_game.empty:
            content_score = content_game['norm_score'].values[0]

        # Get collaborative filtering score if available
        collab_game = collab_recs[collab_recs['app_id'] == game_id]
        if not collab_game.empty:
            collab_score = collab_game['norm_score'].values[0]

        # Calculate weighted hybrid score
        hybrid_score = (content_weight * content_score) + ((1 - content_weight) * collab_score)

        # Skip games the user has already played
        if game_id in user_history.index:
            continue

        # Add to hybrid scores
        hybrid_scores[game_id] = hybrid_score

    # Sort and get top recommendations
    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Create a dataframe
    hybrid_recs = pd.DataFrame(sorted_hybrid, columns=['app_id', 'hybrid_score'])

    # Add game details
    game_details = games_df[games_df['app_id'].isin(hybrid_recs['app_id'])]
    hybrid_recs = hybrid_recs.merge(game_details[['app_id', 'name']], on='app_id')

    return hybrid_recs[['app_id', 'name', 'hybrid_score']]


# --- NEW EVALUATION METRICS USING SIMILARITY SCORES ---
print("\nImplementing similarity-based evaluation metrics...")


def evaluate_recommender_by_similarity(test_data, game_features, user_game_matrix, top_n=10):
    """
    Evaluate recommender system using similarity between recommended items and test items.

    Args:
        test_data: DataFrame with test data
        game_features: DataFrame with game features for calculating similarity
        user_game_matrix: Original user-game interaction matrix
        top_n: Number of recommendations to generate

    Returns:
        avg_similarity: Average similarity score between recommended and actual games
        coverage: Percentage of users for whom recommendations could be generated
        diversity: Average pairwise dissimilarity of recommendations
    """
    # Select a sample of users for evaluation
    test_users = np.random.choice(test_data.index.unique(),
                                  min(100, len(test_data.index.unique())),
                                  replace=False)

    # Initialize metrics
    all_similarities = []
    user_coverage = 0
    diversity_scores = []

    for user_id in test_users:
        # Get the user's actual games from test data
        user_test = test_data.loc[user_id]
        actual_games = user_test[user_test > 0].index.tolist()

        if not actual_games:
            continue

        # Get hybrid recommendations
        recommendations = get_hybrid_recommendations(user_id, top_n=top_n)

        if recommendations.empty:
            continue

        user_coverage += 1
        recommended_games = recommendations['app_id'].tolist()

        # Calculate similarity between recommended games and actual games
        # First, get feature vectors for actual games
        actual_game_indices = [games_df[games_df['app_id'] == game].index[0]
                               for game in actual_games
                               if game in games_df['app_id'].values]

        if not actual_game_indices:
            continue

        actual_features = game_features.iloc[actual_game_indices].values

        # Get feature vectors for recommended games
        rec_game_indices = [games_df[games_df['app_id'] == game].index[0]
                            for game in recommended_games
                            if game in games_df['app_id'].values]

        if not rec_game_indices:
            continue

        rec_features = game_features.iloc[rec_game_indices].values

        # Calculate the average similarity between each recommended game and the closest actual game
        game_similarities = []
        for rec_feature in rec_features:
            # Reshape for cosine_similarity function
            reshaped_rec = rec_feature.reshape(1, -1)
            sim_scores = cosine_similarity(reshaped_rec, actual_features)[0]
            # Take the highest similarity score (closest match)
            best_match_score = np.max(sim_scores)
            game_similarities.append(best_match_score)

        # Average similarity score for this user
        avg_user_similarity = np.mean(game_similarities)
        all_similarities.append(avg_user_similarity)

        # Calculate diversity as the average pairwise dissimilarity between recommendations
        if len(rec_features) > 1:
            pairwise_sim = cosine_similarity(rec_features)
            # Set diagonal elements to 0 to exclude self-similarity
            np.fill_diagonal(pairwise_sim, 0)
            # Convert similarity to dissimilarity (1 - similarity)
            pairwise_dissim = 1 - pairwise_sim
            # Calculate mean dissimilarity
            diversity_score = np.mean(pairwise_dissim)
            diversity_scores.append(diversity_score)

    # Calculate overall metrics
    avg_similarity = np.mean(all_similarities) if all_similarities else 0
    coverage = user_coverage / len(test_users)
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0

    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"User Coverage: {coverage:.4f}")
    print(f"Average Diversity: {avg_diversity:.4f}")

    return avg_similarity, coverage, avg_diversity


def run_enhanced_evaluation(test_data, game_features, user_game_matrix, top_n=10, similarity_threshold=0.5):
    """
    Run both similarity-based and ranking-based evaluation metrics,
    plus precision, recall, and F1 score with similarity threshold.

    Args:
        test_data: DataFrame with test data
        game_features: DataFrame with game features
        user_game_matrix: Original user-game interaction matrix
        top_n: Number of recommendations to generate
        similarity_threshold: Threshold to determine relevance (0-1)
    """
    print("\n--- Similarity-Based Evaluation ---")
    avg_similarity, coverage, diversity = evaluate_recommender_by_similarity(
        test_data, game_features, user_game_matrix, top_n)

    print(f"\n--- Precision, Recall, F1 Evaluation (Similarity Threshold = {similarity_threshold}) ---")
    avg_precision, avg_recall, avg_f1 = evaluate_recommender_precision_recall(
        test_data, user_game_matrix, game_features, k=top_n, similarity_threshold=similarity_threshold)

    # Create a comprehensive report
    print("\n--- Comprehensive Evaluation Report ---")
    print(f"Number of recommendations evaluated: {top_n}")
    print(f"Similarity Score: {avg_similarity:.4f} - How similar recommended games are to user's actual games")
    print(f"User Coverage: {coverage:.4f} - Fraction of users for whom recommendations could be generated")
    print(
        f"Diversity: {diversity:.4f} - How different the recommendations are from each other (higher is more diverse)")
    print(
        f"Precision@{top_n} (similarity threshold {similarity_threshold}): {avg_precision:.4f} - Fraction of recommended items that are relevant")
    print(
        f"Recall@{top_n} (similarity threshold {similarity_threshold}): {avg_recall:.4f} - Fraction of relevant items that are recommended")
    print(
        f"F1@{top_n} (similarity threshold {similarity_threshold}): {avg_f1:.4f} - Harmonic mean of precision and recall")

    # Check if avg_ndcg and avg_mrr are defined - these seem to be missing in your code
    try:
        print(f"NDCG: {avg_ndcg:.4f} - How well the system ranks relevant items")
        print(f"MRR: {avg_mrr:.4f} - Average position of the first relevant recommendation")
    except NameError:
        # The variables don't exist, so skip them
        print("Note: NDCG and MRR metrics were not calculated")

    return {
        'similarity': avg_similarity,
        'coverage': coverage,
        'diversity': diversity,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'similarity_threshold': similarity_threshold
    }


def analyze_test_data(test_data, user_game_matrix):
    """Analyze test data to understand potential issues."""
    # Count number of users with test data
    users_with_test = sum(1 for user_id in test_data.index if test_data.loc[user_id].max() > 0)

    # Count average number of games per user in test set
    games_per_user = [len(test_data.loc[user_id][test_data.loc[user_id] > 0])
                      for user_id in test_data.index]
    avg_games = np.mean(games_per_user) if games_per_user else 0

    # Check data types between recommendations and test data
    sample_user = test_data.index[0]
    sample_test_game = test_data.loc[sample_user][test_data.loc[sample_user] > 0].index[0] if any(
        test_data.loc[sample_user] > 0) else None

    if sample_test_game is not None:
        sample_recs = get_hybrid_recommendations(sample_user, top_n=10)
        if not sample_recs.empty:
            sample_rec_game = sample_recs['app_id'].iloc[0]
            print(f"\nData type analysis:")
            print(f"Test game ID type: {type(sample_test_game)}")
            print(f"Recommended game ID type: {type(sample_rec_game)}")

    print(f"\nTest Data Analysis:")
    print(f"Users with test data: {users_with_test} out of {len(test_data.index)}")
    print(f"Average games per user in test set: {avg_games:.2f}")
    print(f"Test data density: {test_data.values.mean():.6f}")

    # Check for any type conversion issues in game_features
    print(f"\nGame Features Analysis:")
    print(f"game_features data types: {game_features.dtypes.value_counts().to_dict()}")
    print(f"NaN values in game_features: {game_features.isna().sum().sum()}")
    print(f"Game features shape: {game_features.shape}")

# Create test data from the test matrix
print("Converting test matrix to dataframe...")
# Since we're keeping the same shape as the original user_game_matrix
# we need to convert sparse matrix to dense array for dataframe
test_data = pd.DataFrame(
    test_matrix.toarray(),
    index=user_game_matrix.index,
    columns=user_game_matrix.columns
)
print(f"Test data shape: {test_data.shape}")


def calculate_precision_recall_f1_at_k(recommendations, actual_items, k=10):
    """
    Calculate precision, recall, and F1 score at K for recommendations.

    Args:
        recommendations: List of recommended item IDs
        actual_items: List of actual item IDs from test data
        k: Number of recommendations to consider

    Returns:
        precision: Precision@K score
        recall: Recall@K score
        f1: F1@K score
    """
    # Consider only top-K recommendations
    if len(recommendations) > k:
        recommendations = recommendations[:k]

    # Calculate number of relevant items
    n_rel_and_rec = len(set(recommendations) & set(actual_items))

    if len(recommendations) == 0:
        return 0, 0, 0

    # Calculate precision and recall
    precision = n_rel_and_rec / len(recommendations)
    recall = n_rel_and_rec / len(actual_items) if len(actual_items) > 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_precision_recall_f1_at_k_with_similarity(recommendations, actual_items, game_features,
                                                       similarity_threshold=0.5, k=10):
    """
    Calculate precision, recall, and F1 score at K for recommendations using similarity threshold.

    Args:
        recommendations: List of recommended item IDs
        actual_items: List of actual item IDs from test data
        game_features: DataFrame with game features for calculating similarity
        similarity_threshold: Threshold to determine relevance (0-1)
        k: Number of recommendations to consider

    Returns:
        precision: Precision@K score
        recall: Recall@K score
        f1: F1@K score
    """
    # Consider only top-K recommendations
    if len(recommendations) > k:
        recommendations = recommendations[:k]

    if len(recommendations) == 0 or len(actual_items) == 0:
        return 0, 0, 0

    # Get feature vectors for recommended games
    rec_game_indices = [games_df[games_df['app_id'] == game].index[0]
                        for game in recommendations
                        if game in games_df['app_id'].values]

    # Get feature vectors for actual games
    actual_game_indices = [games_df[games_df['app_id'] == game].index[0]
                           for game in actual_items
                           if game in games_df['app_id'].values]

    if not rec_game_indices or not actual_game_indices:
        return 0, 0, 0

    rec_features = game_features.iloc[rec_game_indices].values
    actual_features = game_features.iloc[actual_game_indices].values

    # Count relevant recommendations (those with similarity > threshold to any actual game)
    relevant_recs = 0
    for rec_feature in rec_features:
        reshaped_rec = rec_feature.reshape(1, -1)
        sim_scores = cosine_similarity(reshaped_rec, actual_features)[0]
        if np.max(sim_scores) >= similarity_threshold:
            relevant_recs += 1

    # Count relevant actual items (those with similarity > threshold to any recommendation)
    relevant_actuals = 0
    for actual_feature in actual_features:
        reshaped_actual = actual_feature.reshape(1, -1)
        sim_scores = cosine_similarity(reshaped_actual, rec_features)[0]
        if np.max(sim_scores) >= similarity_threshold:
            relevant_actuals += 1

    # Calculate precision and recall
    precision = relevant_recs / len(rec_features) if len(rec_features) > 0 else 0
    recall = relevant_actuals / len(actual_features) if len(actual_features) > 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def evaluate_recommender_precision_recall(test_data, user_game_matrix, game_features, k=10, num_users=100,
                                          similarity_threshold=0.5):
    """
    Evaluate recommender system using precision, recall, and F1 at K with similarity threshold.

    Args:
        test_data: DataFrame with test data
        user_game_matrix: Original user-game interaction matrix
        game_features: DataFrame with game features for calculating similarity
        k: Number of recommendations to consider
        num_users: Number of users to sample for evaluation
        similarity_threshold: Threshold to determine relevance (0-1)

    Returns:
        avg_precision: Average precision@K score
        avg_recall: Average recall@K score
        avg_f1: Average F1@K score
    """
    # Select a sample of users for evaluation
    test_users = np.random.choice(test_data.index.unique(),
                                  min(num_users, len(test_data.index.unique())),
                                  replace=False)

    # Initialize metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for user_id in test_users:
        # Get the user's actual games from test data
        user_test = test_data.loc[user_id]
        actual_games = user_test[user_test > 0].index.tolist()

        if not actual_games:
            continue

        # Get hybrid recommendations
        recommendations = get_hybrid_recommendations(user_id, top_n=k)

        if recommendations.empty:
            continue

        recommended_games = recommendations['app_id'].tolist()

        # Calculate precision, recall, and F1 with similarity threshold
        precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
            recommended_games, actual_games, game_features, similarity_threshold, k)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    print(f"Precision@{k} (similarity threshold {similarity_threshold}): {avg_precision:.4f}")
    print(f"Recall@{k} (similarity threshold {similarity_threshold}): {avg_recall:.4f}")
    print(f"F1@{k} (similarity threshold {similarity_threshold}): {avg_f1:.4f}")

    return avg_precision, avg_recall, avg_f1


# --- RUN BOTH EVALUATION APPROACHES ---

print("\n--- Running Enhanced Evaluation (Similarity-based) ---")
try:
    evaluation_results = run_enhanced_evaluation(test_data, game_features, user_game_matrix)
except Exception as e:
    print(f"Error during enhanced evaluation: {e}")
    evaluation_results = None

def get_valid_sample_game():
    """Get a valid sample game with a non-null name."""
    for i in range(len(games_df)):
        game_id = games_df['app_id'].iloc[i]
        game_name = games_df['name'].iloc[i]
        if pd.notna(game_name) and pd.notna(game_id):
            return game_id, game_name
    return None, None

# Then replace the example usage code with:
print("\nExample recommendations:")

# Get recommendations for a sample user
sample_user_id = user_game_matrix.index[0]
hybrid_recommendations = get_hybrid_recommendations(sample_user_id, top_n=10)
print(f"\nHybrid recommendations for user {sample_user_id}:")
print(hybrid_recommendations)

# Get content-based recommendations for a sample game with valid name
sample_game_id, sample_game_name = get_valid_sample_game()
if sample_game_id is not None:
    content_recommendations = get_content_recommendations(sample_game_id, top_n=10)
    print(f"\nContent-based recommendations for game {sample_game_name} (ID: {sample_game_id}):")
    print(content_recommendations)
else:
    print("\nCouldn't find a valid sample game for content-based recommendations.")

# --- EXAMPLE USAGE ---
print("\nExample recommendations:")

# Get recommendations for a sample user
sample_user_id = user_game_matrix.index[0]
hybrid_recommendations = get_hybrid_recommendations(sample_user_id, top_n=10)
print(f"\nHybrid recommendations for user {sample_user_id}:")
print(hybrid_recommendations)

# Get content-based recommendations for a sample game
sample_game_id = games_df['app_id'].iloc[0]
sample_game_name = games_df['name'].iloc[0]
content_recommendations = get_content_recommendations(sample_game_id, top_n=10)
print(f"\nContent-based recommendations for game {sample_game_name} (ID: {sample_game_id}):")
print(content_recommendations)
evaluation_results = run_enhanced_evaluation(test_data, game_features, user_game_matrix, top_n=10, similarity_threshold=0.5)
print("\nRecommender system implementation complete!")
