import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class SteamGameRecommender:
    def __init__(self, data_path):
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Load the dataset
        self.df = pd.read_csv(data_path)

        # Basic data cleaning and preprocessing
        self.preprocess_data()

        # Prepare user-game matrix
        self.prepare_user_game_matrix()

    def preprocess_data(self):
        # Remove duplicate entries
        self.df.drop_duplicates(subset=['steam_id', 'app_id'], inplace=True)

        # Ensure numeric columns
        self.df['steam_id'] = pd.to_numeric(self.df['steam_id'], errors='coerce')
        self.df['app_id'] = pd.to_numeric(self.df['app_id'], errors='coerce')
        self.df['playtime_forever'] = pd.to_numeric(self.df['playtime_forever'], errors='coerce')

        # Remove rows with NaN values
        self.df.dropna(subset=['steam_id', 'app_id', 'playtime_forever'], inplace=True)

        # Detailed debugging of playtime
        print("\nDetailed Playtime Analysis:")
        print("Total rows:", len(self.df))

        # Analyze playtime distribution per user
        user_game_counts = self.df.groupby('steam_id').agg({
            'app_id': 'count',
            'playtime_forever': ['count', 'sum', 'mean', 'max']
        })
        user_game_counts.columns = ['total_games', 'games_with_playtime', 'total_playtime', 'mean_playtime',
                                    'max_playtime']

        print("\nUser Game and Playtime Statistics:")
        print(user_game_counts.describe())

        # Filter users with meaningful playtime
        self.df = self.df[self.df['playtime_forever'] > 0]

        print("\nAfter filtering for positive playtime:")
        print("Total rows:", len(self.df))

    def prepare_user_game_matrix(self):
        try:
            # Create a pivot table with steam_id as rows, app_id as columns, and playtime_forever as values
            self.user_game_matrix = self.df.pivot_table(
                index='steam_id',
                columns='app_id',
                values='playtime_forever',
                fill_value=0
            )
            print(f"\nUser-Game Matrix Shape: {self.user_game_matrix.shape}")

            # Analyze matrix characteristics
            print("\nMatrix Characteristics:")
            print("Users with at least one game played:",
                  len(self.user_game_matrix[self.user_game_matrix.sum(axis=1) > 0]))
            print("Unique games played:",
                  len(self.user_game_matrix.columns[self.user_game_matrix.sum() > 0]))
        except Exception as e:
            print(f"Error creating user-game matrix: {e}")
            raise

    def split_data(self, test_size=0.3, random_state=42):
        # Ensure we have enough unique users with playtime
        non_zero_users = self.user_game_matrix[self.user_game_matrix.sum(axis=1) > 0].index

        if len(non_zero_users) < 10:
            raise ValueError("Not enough users with playtime in the dataset")

        # Split the user IDs into train and test sets
        train_ids, test_ids = train_test_split(non_zero_users, test_size=test_size, random_state=random_state)

        # Create train and test matrices
        self.train_matrix = self.user_game_matrix.loc[train_ids]
        self.test_matrix = self.user_game_matrix.loc[test_ids]

        print(f"\nTrain Matrix Shape: {self.train_matrix.shape}")
        print(f"Test Matrix Shape: {self.test_matrix.shape}")

        # Detailed analysis of train and test matrices
        print("\nTrain Matrix Analysis:")
        self.analyze_matrix(self.train_matrix, "Train")

        print("\nTest Matrix Analysis:")
        self.analyze_matrix(self.test_matrix, "Test")

        return self.train_matrix, self.test_matrix

    def analyze_matrix(self, matrix, matrix_name):
        # Users with non-zero total playtime
        non_zero_users = matrix[matrix.sum(axis=1) > 0]
        print(f"{matrix_name} Matrix - Users with Non-Zero Playtime: {len(non_zero_users)}")

        # Games with non-zero total playtime
        non_zero_games = matrix.columns[matrix.sum() > 0]
        print(f"{matrix_name} Matrix - Games with Non-Zero Playtime: {len(non_zero_games)}")

        # Detailed playtime statistics
        playtime_stats = matrix.sum()
        print(f"{matrix_name} Matrix - Playtime Statistics:")
        print(f"  Min Playtime per Game: {playtime_stats.min()}")
        print(f"  Max Playtime per Game: {playtime_stats.max()}")
        print(f"  Mean Playtime per Game: {playtime_stats.mean():.2f}")

        # Print first few non-zero users with their top played games
        print(f"\n{matrix_name} Matrix - Sample Users with Playtime:")
        for user in non_zero_users.index[:5]:
            user_games = matrix.loc[user]
            top_games = user_games[user_games > 0].nlargest(3)
            print(f"User {user}:")
            print(top_games)

    def get_user_games(self, user_id, matrix, min_playtime=0):
        # Get games the user has played (with playtime above threshold)
        return matrix.loc[user_id][matrix.loc[user_id] > min_playtime].index.tolist()

    def recommend_games(self, user_id, n_recommendations=5, min_playtime=0):
        # If user is not in the training matrix, use global recommendations
        if user_id not in self.train_matrix.index:
            return self.global_recommendations(n_recommendations)

        # Get the user's played games
        user_games = self.get_user_games(user_id, self.train_matrix, min_playtime)

        # If no games, return global recommendations
        if not user_games:
            return self.global_recommendations(n_recommendations)

        # Calculate similarity based on playtime
        user_profile = self.train_matrix.loc[user_id]

        # Calculate game similarities
        game_similarities = {}
        for game in self.train_matrix.columns:
            if game not in user_games:
                # Calculate similarity with games the user has played
                try:
                    game_sim = np.mean([
                        cosine_similarity(
                            user_profile[game].reshape(1, -1),
                            user_profile[played_game].reshape(1, -1)
                        )[0][0]
                        for played_game in user_games
                    ])
                    game_similarities[game] = game_sim
                except Exception as e:
                    # Skip games that cause similarity calculation issues
                    continue

        # Sort and get top recommendations
        recommendations = sorted(game_similarities.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

        return [rec[0] for rec in recommendations]

    def global_recommendations(self, n_recommendations=5):
        # Get most played games globally
        global_play_counts = self.train_matrix.sum()
        return global_play_counts.nlargest(n_recommendations).index.tolist()

    def evaluate_recommendations(self, min_playtime=10):
        # Metrics to track recommendation performance
        hit_rates = []

        # Evaluation counters for debugging
        total_users_processed = 0
        users_with_test_games = 0

        # Identify users with playtime above threshold in test matrix
        test_non_zero_users = self.test_matrix[self.test_matrix.sum(axis=1) >= min_playtime].index

        print(f"\nEvaluating recommendations for users with playtime >= {min_playtime}")
        print(f"Number of users to evaluate: {len(test_non_zero_users)}")

        # Evaluate recommendations for users with non-zero playtime
        for user_id in test_non_zero_users:
            total_users_processed += 1

            # Skip users with no games in training set
            if user_id not in self.train_matrix.index:
                continue

            # Get user's games in test set
            test_user_games = self.get_user_games(user_id, self.test_matrix, min_playtime)

            # Skip users with no games in test set
            if not test_user_games:
                continue

            users_with_test_games += 1

            # Get recommendations
            recommendations = self.recommend_games(user_id, min_playtime=min_playtime)

            # Calculate hit rate
            hits = len(set(recommendations) & set(test_user_games))
            hit_rate = hits / len(test_user_games)
            hit_rates.append(hit_rate)

        # Print debugging information
        print(f"Total Users Processed: {total_users_processed}")
        print(f"Users with Test Games: {users_with_test_games}")
        print(f"Hit Rates Calculated: {len(hit_rates)}")

        # Return average hit rate
        if hit_rates:
            return np.mean(hit_rates)
        else:
            print("No hit rates could be calculated.")
            return 0


# Example usage
def main():
    try:
        # Initialize the recommender
        recommender = SteamGameRecommender('steam_games_data.csv')

        # Split the data
        train_matrix, test_matrix = recommender.split_data()

        # Evaluate the recommender
        hit_rate = recommender.evaluate_recommendations()
        print(f"\nAverage Hit Rate: {hit_rate:.2%}")

        # Example recommendations for a specific user
        example_user = test_matrix.index[0]
        recommendations = recommender.recommend_games(example_user)
        print(f"\nRecommendations for User {example_user}:")
        print(recommendations)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()