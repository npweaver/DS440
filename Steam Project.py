import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
user_data = pd.read_csv('final_steam_user_data.csv')
games_data = pd.read_csv('final_cleaned_games.csv')

# Print column names to verify
print("User data columns:", user_data.columns.tolist())
print("Games data columns:", games_data.columns.tolist())

# Handle column name discrepancies
# Rename columns if needed to ensure consistency
if 'appid' in games_data.columns and 'app_id' not in games_data.columns:
    games_data = games_data.rename(columns={'appid': 'app_id'})
    print("Renamed 'appid' to 'app_id' in games_data")

# Examine the first few rows to understand the data
print("\nUser data sample:")
print(user_data.head())
print("\nGames data sample:")
print(games_data.head())

# Create a user-item matrix where rows are users and columns are games
print("\nCreating user-item matrix...")
# This creates a matrix where each cell indicates if a user owns a game (1) or not (0)
user_game_matrix = user_data.pivot_table(
    index='steam_id',
    columns='app_id',
    values='name',  # Using 'name' as a placeholder; we only care if a value exists
    aggfunc=lambda x: 1,  # Set to 1 if the user owns the game
    fill_value=0  # 0 if they don't own it
)

# Print the shape of the matrix
print(f"Matrix shape: {user_game_matrix.shape}")

# Convert to sparse matrix for efficiency (many cells will be 0)
user_game_sparse = csr_matrix(user_game_matrix.values)

# Initialize the model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(user_game_sparse)


def get_game_recommendations(user_id, n_recommendations=10):
    """
    Get game recommendations for a specific user

    Parameters:
    -----------
    user_id : int
        The steam_id of the user
    n_recommendations : int
        Number of recommendations to return

    Returns:
    --------
    DataFrame containing recommended games
    """
    # Check if user exists in our data
    if user_id not in user_game_matrix.index:
        return f"User {user_id} not found in the dataset."

    # Find the index of the user
    user_idx = user_game_matrix.index.get_loc(user_id)

    # Get the user's profile
    user_profile = user_game_sparse[user_idx:user_idx + 1]

    # Find similar users
    distances, indices = model_knn.kneighbors(user_profile, n_neighbors=n_recommendations + 1)

    # Get the indices of similar users (skip the first one as it's the user themselves)
    similar_users_indices = indices.flatten()[1:]
    similar_users = user_game_matrix.iloc[similar_users_indices]

    # Convert back to user IDs
    similar_users_ids = user_game_matrix.index[similar_users_indices]
    print(f"Similar users to {user_id}: {similar_users_ids.tolist()}")

    # Get the games owned by the user
    user_games = set(user_game_matrix.columns[user_game_matrix.loc[user_id] > 0])

    # Create a dictionary to store game scores
    game_scores = {}

    # Calculate scores for each game
    for game in user_game_matrix.columns:
        # Skip games the user already owns
        if game in user_games:
            continue

        # Calculate a simple score based on how many similar users own the game
        game_idx = user_game_matrix.columns.get_loc(game)
        score = similar_users.iloc[:, game_idx].sum()

        if score > 0:
            game_scores[game] = score

    # Sort games by score
    recommended_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Get game details
    recommendations = []
    for game_id, score in recommended_games:
        game_info = games_data[games_data['app_id'] == game_id]
        if not game_info.empty:
            recommendations.append({
                'app_id': game_id,
                'name': game_info['name'].iloc[0] if 'name' in game_info else f"Game {game_id}",
                'score': score,
                'similarity_score': score / len(similar_users),  # Normalize score
                'genre': game_info['genre'].iloc[0] if 'genre' in game_info else "Unknown",
                'tags': game_info['tags'].iloc[0] if 'tags' in game_info else "Unknown"
            })

    return pd.DataFrame(recommendations) if recommendations else "No recommendations found."


# Function to get content-based recommendations using game tags and genres
def get_content_based_recommendations(user_id, n_recommendations=10):
    """
    Get content-based recommendations based on genres and tags of owned games

    Parameters:
    -----------
    user_id : int
        The steam_id of the user
    n_recommendations : int
        Number of recommendations to return

    Returns:
    --------
    DataFrame containing recommended games
    """
    # Check if user exists in our data
    if user_id not in user_game_matrix.index:
        return f"User {user_id} not found in the dataset."

    # Get the games owned by the user
    user_games = user_game_matrix.columns[user_game_matrix.loc[user_id] > 0].tolist()

    # Get genres and tags of owned games
    owned_games_info = user_data[(user_data['steam_id'] == user_id) & (user_data['app_id'].isin(user_games))]
    owned_game_ids = owned_games_info['app_id'].unique()

    # Get full game details from games_data
    user_game_details = games_data[games_data['app_id'].isin(owned_game_ids)]

    # Extract and combine all tags and genres
    all_tags = []
    all_genres = []

    for _, game in user_game_details.iterrows():
        if 'tags' in games_data.columns and isinstance(game.get('tags'), str):
            all_tags.extend(game['tags'].split(','))
        if 'genre' in games_data.columns and isinstance(game.get('genre'), str):
            all_genres.extend(game['genre'].split(','))

    # Count tag and genre occurrences
    tag_counts = pd.Series(all_tags).value_counts()
    genre_counts = pd.Series(all_genres).value_counts()

    # Get the top tags and genres
    top_tags = tag_counts.index[:min(10, len(tag_counts))].tolist() if not tag_counts.empty else []
    top_genres = genre_counts.index[:min(5, len(genre_counts))].tolist() if not genre_counts.empty else []

    print(f"User's top genres: {top_genres}")
    print(f"User's top tags: {top_tags[:5] if len(top_tags) >= 5 else top_tags}")

    # Score all games based on tag and genre similarity
    game_scores = {}

    for _, game in games_data.iterrows():
        game_id = game['app_id']

        # Skip games the user already owns
        if game_id in owned_game_ids:
            continue

        score = 0
        # Score based on genres
        if 'genre' in games_data.columns and isinstance(game.get('genre'), str):
            game_genres = game['genre'].split(',')
            genre_match = sum(genre in top_genres for genre in game_genres)
            score += genre_match * 2  # Weight genres more heavily

        # Score based on tags
        if 'tags' in games_data.columns and isinstance(game.get('tags'), str):
            game_tags = game['tags'].split(',')
            tag_match = sum(tag in top_tags for tag in game_tags)
            score += tag_match

        if score > 0:
            game_scores[game_id] = score

    # Sort games by score
    recommended_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Get game details
    recommendations = []
    for game_id, score in recommended_games:
        game_info = games_data[games_data['app_id'] == game_id]
        if not game_info.empty:
            recommendations.append({
                'app_id': game_id,
                'name': game_info['name'].iloc[0] if 'name' in game_info else f"Game {game_id}",
                'score': score,
                'genre': game_info['genre'].iloc[0] if 'genre' in game_info else "Unknown",
                'tags': game_info['tags'].iloc[0] if 'tags' in game_info else "Unknown"
            })

    return pd.DataFrame(recommendations) if recommendations else "No content-based recommendations found."


# Function to combine both recommendation approaches
def hybrid_recommendations(user_id, n_recommendations=10):
    """
    Combine collaborative filtering and content-based recommendations

    Parameters:
    -----------
    user_id : int
        The steam_id of the user
    n_recommendations : int
        Number of recommendations to return

    Returns:
    --------
    DataFrame containing recommended games
    """
    # Get collaborative filtering recommendations
    cf_recommendations = get_game_recommendations(user_id, n_recommendations)

    # Get content-based recommendations
    cb_recommendations = get_content_based_recommendations(user_id, n_recommendations)

    # If either recommendation system failed or returned a string, return the other
    if isinstance(cf_recommendations, str):
        return cb_recommendations if not isinstance(cb_recommendations, str) else "No recommendations available."
    if isinstance(cb_recommendations, str):
        return cf_recommendations

    # If both are empty DataFrames
    if cf_recommendations.empty and cb_recommendations.empty:
        return "No recommendations available."

    # If one is empty, return the other
    if cf_recommendations.empty:
        return cb_recommendations
    if cb_recommendations.empty:
        return cf_recommendations

    # Combine scores from both methods
    cf_recommendations = cf_recommendations.set_index('app_id')
    cb_recommendations = cb_recommendations.set_index('app_id')

    # Normalize scores for each method
    if not cf_recommendations.empty and 'score' in cf_recommendations.columns:
        cf_recommendations['norm_score'] = cf_recommendations['score'] / cf_recommendations['score'].max() if \
        cf_recommendations['score'].max() > 0 else 0
    if not cb_recommendations.empty and 'score' in cb_recommendations.columns:
        cb_recommendations['norm_score'] = cb_recommendations['score'] / cb_recommendations['score'].max() if \
        cb_recommendations['score'].max() > 0 else 0

    # Combine recommendations
    all_app_ids = set(cf_recommendations.index) | set(cb_recommendations.index)

    # Create the combined DataFrame
    combined_recommendations = []

    for app_id in all_app_ids:
        # Calculate hybrid score
        cf_score = cf_recommendations.loc[
            app_id, 'norm_score'] if app_id in cf_recommendations.index and 'norm_score' in cf_recommendations.columns else 0
        cb_score = cb_recommendations.loc[
            app_id, 'norm_score'] if app_id in cb_recommendations.index and 'norm_score' in cb_recommendations.columns else 0

        # Weight collaborative filtering slightly more (0.6 vs 0.4)
        hybrid_score = (0.6 * cf_score) + (0.4 * cb_score)

        # Get game details
        game_df = cf_recommendations.loc[[app_id]] if app_id in cf_recommendations.index else cb_recommendations.loc[
            [app_id]]

        game_info = {
            'app_id': app_id,
            'name': games_data[games_data['app_id'] == app_id]['name'].iloc[0] if not games_data[
                games_data['app_id'] == app_id].empty and 'name' in games_data.columns else f"Game {app_id}",
            'hybrid_score': hybrid_score,
            'cf_score': cf_score,
            'cb_score': cb_score
        }

        # Add genre and tags if available
        if 'genre' in game_df.columns:
            game_info['genre'] = game_df['genre'].iloc[0]
        if 'tags' in game_df.columns:
            game_info['tags'] = game_df['tags'].iloc[0]

        combined_recommendations.append(game_info)

    # Convert to DataFrame and sort by hybrid score
    recommendations_df = pd.DataFrame(combined_recommendations)
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values('hybrid_score', ascending=False).head(n_recommendations)

    return recommendations_df.reset_index(
        drop=True) if not recommendations_df.empty else "No hybrid recommendations available."


# Demo: How to use the recommender system
if __name__ == "__main__":
    # Let's find a user with a reasonable number of games
    user_game_counts = user_data.groupby('steam_id').size()
    users_with_many_games = user_game_counts[user_game_counts > 10].index[:5]

    print(f"\nTop users with many games: {users_with_many_games.tolist()}")

    # Select the first user from the list for demonstration
    if len(users_with_many_games) > 0:
        demo_user_id = users_with_many_games[0]

        print(f"\n--- DEMO FOR USER {demo_user_id} ---")

        try:
            # Get collaborative filtering recommendations
            print("\nCollaborative filtering recommendations:")
            cf_recs = get_game_recommendations(demo_user_id)
            if isinstance(cf_recs, pd.DataFrame) and not cf_recs.empty:
                selected_columns = [col for col in ['name', 'similarity_score', 'genre'] if col in cf_recs.columns]
                print(cf_recs[selected_columns])
            else:
                print(cf_recs)

            # Get content-based recommendations
            print("\nContent-based recommendations:")
            cb_recs = get_content_based_recommendations(demo_user_id)
            if isinstance(cb_recs, pd.DataFrame) and not cb_recs.empty:
                selected_columns = [col for col in ['name', 'score', 'genre'] if col in cb_recs.columns]
                print(cb_recs[selected_columns])
            else:
                print(cb_recs)

            # Get hybrid recommendations
            print("\nHybrid recommendations:")
            hybrid_recs = hybrid_recommendations(demo_user_id)
            if isinstance(hybrid_recs, pd.DataFrame) and not hybrid_recs.empty:
                selected_columns = [col for col in ['name', 'hybrid_score', 'cf_score', 'cb_score', 'genre'] if
                                    col in hybrid_recs.columns]
                print(hybrid_recs[selected_columns])
            else:
                print(hybrid_recs)

            # Show owned games for reference
            user_owned_games = user_data[user_data['steam_id'] == demo_user_id]
            print(f"\nUser owns {len(user_owned_games)} games, including:")
            print(user_owned_games['name'].head(10).tolist())

        except Exception as e:
            print(f"Error in recommendation process: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("No users with enough games found for a demo.")