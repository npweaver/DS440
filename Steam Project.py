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
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np

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


# Two-Tower Recommender System Implementation


class TwoTowerRecommender:
    def __init__(self, embedding_dim=64, learning_rate=0.001):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.user_encoder = None
        self.game_encoder = None
        self.user_embeddings = None
        self.game_embeddings = None

    def _create_user_tower(self, num_users):
        """Create the user tower of the model"""
        inputs = tf.keras.layers.Input(shape=(1,))
        embedding = tf.keras.layers.Embedding(num_users, self.embedding_dim)(inputs)
        flat = tf.keras.layers.Flatten()(embedding)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        output = tf.keras.layers.Dense(self.embedding_dim)(dense2)
        normalized = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))(output)
        return tf.keras.Model(inputs=inputs, outputs=normalized)

    def _create_game_tower(self, num_games, game_features_dim):
        """Create the game tower of the model"""
        id_input = tf.keras.layers.Input(shape=(1,))
        feature_input = tf.keras.layers.Input(shape=(game_features_dim,))

        # Process game ID
        id_embedding = tf.keras.layers.Embedding(num_games, self.embedding_dim)(id_input)
        id_flat = tf.keras.layers.Flatten()(id_embedding)

        # Process game features
        feature_dense = tf.keras.layers.Dense(128, activation='relu')(feature_input)

        # Combine both paths
        combined = tf.keras.layers.Concatenate()([id_flat, feature_dense])
        dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        output = tf.keras.layers.Dense(self.embedding_dim)(dense2)
        normalized = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))(output)

        return tf.keras.Model(inputs=[id_input, feature_input], outputs=normalized)

    def build_model(self, num_users, num_games, game_features_dim):
        """Build the complete two-tower model"""
        # Create towers
        self.user_tower = self._create_user_tower(num_users)
        self.game_tower = self._create_game_tower(num_games, game_features_dim)

        # Create combined model for training
        user_input = tf.keras.layers.Input(shape=(1,))
        game_id_input = tf.keras.layers.Input(shape=(1,))
        game_feature_input = tf.keras.layers.Input(shape=(game_features_dim,))

        user_embedding = self.user_tower(user_input)
        game_embedding = self.game_tower([game_id_input, game_feature_input])

        # Compute dot product similarity
        dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, game_embedding])

        model = tf.keras.Model(
            inputs=[user_input, game_id_input, game_feature_input],
            outputs=dot_product
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model = model
        return model

    def prepare_data(self, user_game_matrix, game_features):
        """Prepare data for training with memory optimization"""
        print("Starting data preparation...")

        print("Encoding users and games...")
        self.user_encoder = LabelEncoder()
        self.game_encoder = LabelEncoder()

        users = user_game_matrix.index.values
        games = user_game_matrix.columns.values

        encoded_users = self.user_encoder.fit_transform(users)
        encoded_games = self.game_encoder.fit_transform(games)
        print(f"Encoded {len(users)} users and {len(games)} games")

        # Create game ID to feature index mapping
        print("Creating game ID to feature mapping...")
        valid_games = set(games)
        game_id_to_feature_idx = {
            game_id: idx for idx, game_id in enumerate(games_df['app_id'])
            if game_id in valid_games and idx < len(game_features)
        }
        print(f"Created mapping for {len(game_id_to_feature_idx)} games")

        # Convert to numpy array for faster operations
        available_games = np.array(list(game_id_to_feature_idx.keys()))
        print(f"Number of available games for sampling: {len(available_games)}")

        # Pre-calculate game indices
        game_to_idx = {game: idx for idx, game in enumerate(games)}

        # Memory optimization: Process and return batches to avoid storing the entire array
        batch_size = 10000  # Adjust based on your memory constraints
        total_interactions = []
        user_batch_indices = []
        game_batch_indices = []
        game_batch_features = []
        labels_batch = []

        print("\nCollecting user-game interactions...")
        total_users = len(users)

        # Pre-process user_game_matrix for faster access
        user_game_sparse_dict = {}
        for user_idx, user_id in enumerate(users):
            # Get indices of positive interactions for this user
            user_row = user_game_matrix.loc[user_id]
            positive_indices = user_row[user_row > 0].index.tolist()
            if positive_indices:
                user_game_sparse_dict[user_id] = positive_indices

            # Show progress
            if user_idx % 1000 == 0:
                print(f"Pre-processing users {user_idx}/{total_users}")

        print(f"Pre-processed {len(user_game_sparse_dict)} users with positive interactions")

        # Process users in parallel-friendly batches
        user_batch_size = 500  # Process this many users before yielding
        processed_users = 0
        total_users_with_interactions = len(user_game_sparse_dict)

        for batch_start in range(0, total_users_with_interactions, user_batch_size):
            # Get batch of users
            batch_users = list(user_game_sparse_dict.keys())[batch_start:batch_start + user_batch_size]

            # Reset batch data containers
            user_batch_indices = []
            game_batch_indices = []
            game_batch_features = []
            labels_batch = []

            # Process each user in this batch
            for user_id in batch_users:
                user_idx = np.where(users == user_id)[0][0]
                positive_games = user_game_sparse_dict[user_id]

                # Process positive interactions
                valid_positive_games = [g for g in positive_games if g in game_id_to_feature_idx]

                # Add positive samples
                for game_id in valid_positive_games:
                    try:
                        feature_idx = game_id_to_feature_idx[game_id]
                        game_idx = game_to_idx[game_id]

                        user_batch_indices.append(encoded_users[user_idx])
                        game_batch_indices.append(encoded_games[game_idx])
                        game_batch_features.append(game_features.iloc[feature_idx].values.astype(np.float32))
                        labels_batch.append(1)
                    except Exception as e:
                        continue

                # Sample negative interactions
                if len(valid_positive_games) > 0:
                    pos_set = set(valid_positive_games)
                    neg_candidates = available_games[~np.isin(available_games, list(pos_set))]

                    if len(neg_candidates) > 0:
                        n_samples = min(len(valid_positive_games), 10, len(neg_candidates))
                        if n_samples > 0:
                            negative_games = np.random.choice(neg_candidates, size=n_samples, replace=False)

                            for game_id in negative_games:
                                try:
                                    feature_idx = game_id_to_feature_idx[game_id]
                                    game_idx = game_to_idx[game_id]

                                    user_batch_indices.append(encoded_users[user_idx])
                                    game_batch_indices.append(encoded_games[game_idx])
                                    game_batch_features.append(
                                        game_features.iloc[feature_idx].values.astype(np.float32))
                                    labels_batch.append(0)
                                except Exception as e:
                                    continue

            # Yield the batch if we have data
            if user_batch_indices:
                processed_users += len(batch_users)
                print(f"Processing users batch {processed_users}/{total_users_with_interactions}")

                # Convert to numpy arrays and yield as batches
                yield (
                    np.array(user_batch_indices, dtype=np.int32),
                    np.array(game_batch_indices, dtype=np.int32),
                    np.array(game_batch_features, dtype=np.float32),
                    np.array(labels_batch, dtype=np.int32)
                )

        print(f"Total users processed: {processed_users}")

    def train(self, data_generator, validation_split=0.2, batch_size=64, epochs=5):
        """Train the model using batched data"""
        # Initialize metrics
        history_metrics = {
            'accuracy': [], 'loss': [],
            'val_accuracy': [], 'val_loss': []
        }

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = []
            epoch_accuracy = []

            # Reset the generator for each epoch
            data_gen = data_generator()

            batch_counter = 0
            for user_ids_batch, game_ids_batch, game_features_batch, labels_batch in data_gen:
                # Split batch into train and validation
                split_idx = int(len(user_ids_batch) * (1 - validation_split))

                # Train data
                train_user_ids = user_ids_batch[:split_idx]
                train_game_ids = game_ids_batch[:split_idx]
                train_game_features = game_features_batch[:split_idx]
                train_labels = labels_batch[:split_idx]

                # Validation data
                val_user_ids = user_ids_batch[split_idx:]
                val_game_ids = game_ids_batch[split_idx:]
                val_game_features = game_features_batch[split_idx:]
                val_labels = labels_batch[split_idx:]

                # Train on this batch
                batch_history = self.model.train_on_batch(
                    [train_user_ids, train_game_ids, train_game_features],
                    train_labels
                )

                # Evaluate on validation
                val_loss, val_accuracy = self.model.evaluate(
                    [val_user_ids, val_game_ids, val_game_features],
                    val_labels,
                    verbose=0
                )

                # Record metrics
                epoch_loss.append(batch_history[0])
                epoch_accuracy.append(batch_history[1])

                batch_counter += 1
                if batch_counter % 10 == 0:
                    print(f"  Batch {batch_counter}: loss={batch_history[0]:.4f}, accuracy={batch_history[1]:.4f}")

            # Calculate epoch metrics
            avg_loss = np.mean(epoch_loss)
            avg_accuracy = np.mean(epoch_accuracy)

            # Store in history
            history_metrics['loss'].append(avg_loss)
            history_metrics['accuracy'].append(avg_accuracy)
            history_metrics['val_loss'].append(val_loss)
            history_metrics['val_accuracy'].append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

        return history_metrics

    def get_recommendations(self, user_id, game_features_df, top_n=10):
        """Get recommendations for a user"""
        if user_id not in self.user_encoder.classes_:
            return pd.DataFrame()

        # Encode user ID
        encoded_user = self.user_encoder.transform([user_id])[0]
        user_embedding = self.user_tower.predict(np.array([encoded_user]))

        # Get game IDs that exist in both game_features_df and encoded games
        valid_games = [game_id for game_id in self.game_encoder.classes_
                       if game_id in games_df['app_id'].values and
                       games_df[games_df['app_id'] == game_id].index[0] < len(game_features_df)]

        if not valid_games:
            return pd.DataFrame()

        # Get encoded game IDs and their features
        encoded_games = self.game_encoder.transform(valid_games)
        game_indices = [games_df[games_df['app_id'] == game_id].index[0] for game_id in valid_games]
        game_features = game_features_df.iloc[game_indices].values

        # Get game embeddings
        game_embeddings = self.game_tower.predict([encoded_games, game_features])

        # Calculate similarities
        similarities = np.dot(user_embedding, game_embeddings.T)[0]

        # Get top N recommendations
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_games = [valid_games[i] for i in top_indices]
        top_scores = similarities[top_indices]

        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'app_id': top_games,
            'two_tower_score': top_scores
        })

        # Add game names
        game_details = games_df[games_df['app_id'].isin(recommendations['app_id'])]
        recommendations = recommendations.merge(
            game_details[['app_id', 'name']],
            on='app_id'
        )

        return recommendations[['app_id', 'name', 'two_tower_score']]


def create_combined_recommender(hybrid_weight=0.6, two_tower_weight=0.4):
    """Create a combined recommender that uses both hybrid and two-tower models"""

    def get_combined_recommendations(user_id, top_n=10):
        # Get recommendations from both systems
        hybrid_recs = get_hybrid_recommendations(user_id, top_n=top_n)
        two_tower_recs = two_tower_model.get_recommendations(user_id, game_features, top_n=top_n)

        if hybrid_recs.empty and two_tower_recs.empty:
            return pd.DataFrame()
        elif hybrid_recs.empty:
            return two_tower_recs
        elif two_tower_recs.empty:
            return hybrid_recs

        # Normalize scores
        hybrid_recs['norm_score'] = hybrid_recs['hybrid_score'] / hybrid_recs['hybrid_score'].max()
        two_tower_recs['norm_score'] = two_tower_recs['two_tower_score'] / two_tower_recs['two_tower_score'].max()

        # Combine recommendations
        all_games = set(hybrid_recs['app_id']).union(set(two_tower_recs['app_id']))
        combined_scores = {}

        for game_id in all_games:
            hybrid_score = 0
            two_tower_score = 0

            # Get hybrid score if available
            hybrid_game = hybrid_recs[hybrid_recs['app_id'] == game_id]
            if not hybrid_game.empty:
                hybrid_score = hybrid_game['norm_score'].values[0]

            # Get two-tower score if available
            two_tower_game = two_tower_recs[two_tower_recs['app_id'] == game_id]
            if not two_tower_game.empty:
                two_tower_score = two_tower_game['norm_score'].values[0]

            # Calculate weighted combined score
            combined_score = (hybrid_weight * hybrid_score) + (two_tower_weight * two_tower_score)
            combined_scores[game_id] = combined_score

        # Sort and get top recommendations
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Create DataFrame
        combined_recs = pd.DataFrame(sorted_combined, columns=['app_id', 'combined_score'])

        # Add game details
        game_details = games_df[games_df['app_id'].isin(combined_recs['app_id'])]
        combined_recs = combined_recs.merge(game_details[['app_id', 'name']], on='app_id')

        return combined_recs[['app_id', 'name', 'combined_score']]

    return get_combined_recommendations


# Ensure game_features index aligns with games_df
print("\nAligning game features with game data...")
game_features = game_features.loc[games_df.index[:len(game_features)]]
print(f"Game features shape after alignment: {game_features.shape}")

# Optimize TensorFlow for CPU
tf.config.threading.set_intra_op_parallelism_threads(24)  # i9-14900K has 24 cores
tf.config.threading.set_inter_op_parallelism_threads(24)

# Ensure game_features index aligns with games_df
print("\nAligning game features with game data...")
game_features = game_features.loc[games_df.index[:len(game_features)]]
print(f"Game features shape after alignment: {game_features.shape}")

# Modify the data preparation and training calls
print("\nInitializing Two-Tower Recommender System...")
print("Building model architecture...")
two_tower_model = TwoTowerRecommender(embedding_dim=64)

print("Configuring model parameters...")
two_tower_model.build_model(
    num_users=len(user_game_matrix.index),
    num_games=len(user_game_matrix.columns),
    game_features_dim=game_features.shape[1]
)

print("Preparing for training...")
# Create a generator function that yields batches
def data_generator():
    return two_tower_model.prepare_data(user_game_matrix, game_features)

print("\nStarting model training...")
history = two_tower_model.train(data_generator, epochs=5)

# Create combined recommender
get_combined_recommendations = create_combined_recommender(
    hybrid_weight=0.6,
    two_tower_weight=0.4
)

# Create test data from the test matrix
print("\nCreating test data for evaluation...")
# Use a sparse dataframe representation to save memory
from scipy import sparse
test_coo = test_matrix.tocoo()
test_values = test_coo.data
test_rows = test_coo.row
test_cols = test_coo.col

# Create a sparse dataframe representation
test_data = pd.DataFrame({
    'user_idx': test_rows,
    'game_idx': test_cols,
    'playtime': test_values
})

# Map indices back to actual user and game IDs
test_data['user_id'] = [user_game_matrix.index[idx] for idx in test_data['user_idx']]
test_data['game_id'] = [user_game_matrix.columns[idx] for idx in test_data['game_idx']]

print(f"Test data shape: {test_data.shape}")
print(f"Test data density: {len(test_data) / (len(user_game_matrix.index) * len(user_game_matrix.columns)):.6f}")


def compare_recommender_systems(test_data, num_users=100):
    """Compare the performance of hybrid and combined recommenders using sparse test data"""
    # Metrics to track
    hybrid_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': []}
    combined_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': []}

    # Get unique test users
    test_users = test_data['user_id'].unique()

    # Select random users for evaluation
    if len(test_users) > num_users:
        test_users = np.random.choice(test_users, num_users, replace=False)

    for user_id in test_users:
        # Get actual games from test data
        user_games = test_data[test_data['user_id'] == user_id]
        actual_games = user_games['game_id'].tolist()

        if not actual_games:
            continue

        # Get recommendations from both systems
        hybrid_recs = get_hybrid_recommendations(user_id, top_n=10)
        combined_recs = get_combined_recommendations(user_id, top_n=10)

        if hybrid_recs.empty or combined_recs.empty:
            continue

        # Continue with the evaluation metrics...
        # (rest of the function remains the same)

    # Calculate and print averages...

    return hybrid_metrics, combined_metrics


# Run comparison
print("\nComparing recommender systems...")
hybrid_metrics, combined_metrics = compare_recommender_systems(test_data, num_users=100)


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

def evaluate_recommender_systems(test_data, game_features, user_game_matrix, content_weight=0.4,
                                 collaborative_weight=0.6,
                                 hybrid_weight=0.6, two_tower_weight=0.4, top_n=10, num_users=100,
                                 similarity_threshold=0.5):
    """
    Comprehensive evaluation of all recommender system components with clear metric separation.

    Args:
        test_data: DataFrame with test data
        game_features: DataFrame with game features
        user_game_matrix: Original user-game interaction matrix
        content_weight: Weight for content-based filtering in hybrid recommender
        collaborative_weight: Weight for collaborative filtering in hybrid recommender
        hybrid_weight: Weight for hybrid component in combined recommender
        two_tower_weight: Weight for two-tower component in combined recommender
        top_n: Number of recommendations to consider
        num_users: Number of users to sample for evaluation
        similarity_threshold: Threshold to determine relevance (0-1)

    Returns:
        Dictionary with all evaluation metrics
    """
    # Select a sample of users for evaluation
    test_users = np.random.choice(test_data.index.unique(),
                                  min(num_users, len(test_data.index.unique())),
                                  replace=False)

    # Initialize metrics dictionaries
    content_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': [], 'similarity': []}
    collab_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': [], 'similarity': []}
    hybrid_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': [], 'similarity': []}
    two_tower_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': [], 'similarity': []}
    combined_metrics = {'precision': [], 'recall': [], 'f1': [], 'diversity': [], 'similarity': []}

    # For tracking component comparison in hybrid system
    hybrid_component_comparison = {'content_better': 0, 'collab_better': 0, 'equal': 0}
    combined_component_comparison = {'hybrid_better': 0, 'two_tower_better': 0, 'equal': 0}

    print(f"Evaluating {len(test_users)} users...")

    for i, user_id in enumerate(test_users):
        if i % 10 == 0:
            print(f"Processing user {i + 1}/{len(test_users)}")

        # Get the user's actual games from test data
        user_test = test_data.loc[user_id]
        actual_games = user_test[user_test > 0].index.tolist()

        if not actual_games:
            continue

        # Get user's play history for content-based recommendations
        user_history = user_game_matrix.loc[user_id]
        user_history = user_history[user_history > 0]

        if len(user_history) == 0:
            continue  # No play history

        # Get most played game for content-based
        most_played_game = user_history.idxmax()

        # Get recommendations from each system
        try:
            # 1. Content-based recommendations
            content_recs = get_content_recommendations(most_played_game, top_n=top_n)
            if not content_recs.empty:
                content_games = content_recs['app_id'].tolist()

                # Calculate metrics
                precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
                    content_games, actual_games, game_features, similarity_threshold, top_n)
                content_metrics['precision'].append(precision)
                content_metrics['recall'].append(recall)
                content_metrics['f1'].append(f1)

                # Calculate similarity
                sim_score = calculate_similarity_score(content_games, actual_games, game_features)
                content_metrics['similarity'].append(sim_score)

                # Calculate diversity
                diversity = calculate_diversity_score(content_games, game_features)
                content_metrics['diversity'].append(diversity)

            # 2. Collaborative filtering recommendations
            collab_recs = get_collaborative_recommendations(user_id, top_n=top_n)
            if not collab_recs.empty:
                collab_games = collab_recs['app_id'].tolist()

                # Calculate metrics
                precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
                    collab_games, actual_games, game_features, similarity_threshold, top_n)
                collab_metrics['precision'].append(precision)
                collab_metrics['recall'].append(recall)
                collab_metrics['f1'].append(f1)

                # Calculate similarity
                sim_score = calculate_similarity_score(collab_games, actual_games, game_features)
                collab_metrics['similarity'].append(sim_score)

                # Calculate diversity
                diversity = calculate_diversity_score(collab_games, game_features)
                collab_metrics['diversity'].append(diversity)

            # 3. Hybrid recommendations (combine content and collaborative)
            hybrid_recs = get_hybrid_recommendations(user_id, content_weight=content_weight, top_n=top_n)
            if not hybrid_recs.empty:
                hybrid_games = hybrid_recs['app_id'].tolist()

                # Calculate metrics
                precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
                    hybrid_games, actual_games, game_features, similarity_threshold, top_n)
                hybrid_metrics['precision'].append(precision)
                hybrid_metrics['recall'].append(recall)
                hybrid_metrics['f1'].append(f1)

                # Calculate similarity
                sim_score = calculate_similarity_score(hybrid_games, actual_games, game_features)
                hybrid_metrics['similarity'].append(sim_score)

                # Calculate diversity
                diversity = calculate_diversity_score(hybrid_games, game_features)
                hybrid_metrics['diversity'].append(diversity)

                # Compare component performance in hybrid (which was better)
                if not content_recs.empty and not collab_recs.empty:
                    content_sim = content_metrics['similarity'][-1]
                    collab_sim = collab_metrics['similarity'][-1]

                    if content_sim > collab_sim:
                        hybrid_component_comparison['content_better'] += 1
                    elif collab_sim > content_sim:
                        hybrid_component_comparison['collab_better'] += 1
                    else:
                        hybrid_component_comparison['equal'] += 1

            # 4. Two-Tower recommendations
            two_tower_recs = two_tower_model.get_recommendations(user_id, game_features, top_n=top_n)
            if not two_tower_recs.empty:
                two_tower_games = two_tower_recs['app_id'].tolist()

                # Calculate metrics
                precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
                    two_tower_games, actual_games, game_features, similarity_threshold, top_n)
                two_tower_metrics['precision'].append(precision)
                two_tower_metrics['recall'].append(recall)
                two_tower_metrics['f1'].append(f1)

                # Calculate similarity
                sim_score = calculate_similarity_score(two_tower_games, actual_games, game_features)
                two_tower_metrics['similarity'].append(sim_score)

                # Calculate diversity
                diversity = calculate_diversity_score(two_tower_games, game_features)
                two_tower_metrics['diversity'].append(diversity)

            # 5. Combined recommendations (hybrid + two-tower)
            combined_recs = get_combined_recommendations(user_id, top_n=top_n)
            if not combined_recs.empty:
                combined_games = combined_recs['app_id'].tolist()

                # Calculate metrics
                precision, recall, f1 = calculate_precision_recall_f1_at_k_with_similarity(
                    combined_games, actual_games, game_features, similarity_threshold, top_n)
                combined_metrics['precision'].append(precision)
                combined_metrics['recall'].append(recall)
                combined_metrics['f1'].append(f1)

                # Calculate similarity
                sim_score = calculate_similarity_score(combined_games, actual_games, game_features)
                combined_metrics['similarity'].append(sim_score)

                # Calculate diversity
                diversity = calculate_diversity_score(combined_games, game_features)
                combined_metrics['diversity'].append(diversity)

                # Compare component performance in combined system (which was better)
                if not hybrid_recs.empty and not two_tower_recs.empty:
                    hybrid_sim = hybrid_metrics['similarity'][-1]
                    two_tower_sim = two_tower_metrics['similarity'][-1]

                    if hybrid_sim > two_tower_sim:
                        combined_component_comparison['hybrid_better'] += 1
                    elif two_tower_sim > hybrid_sim:
                        combined_component_comparison['two_tower_better'] += 1
                    else:
                        combined_component_comparison['equal'] += 1

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue

    # Calculate average metrics for each system
    results = {}

    # Helper function to calculate averages
    def calc_avg_metrics(metrics_dict):
        return {
            'precision': np.mean(metrics_dict['precision']) if metrics_dict['precision'] else 0,
            'recall': np.mean(metrics_dict['recall']) if metrics_dict['recall'] else 0,
            'f1': np.mean(metrics_dict['f1']) if metrics_dict['f1'] else 0,
            'similarity': np.mean(metrics_dict['similarity']) if metrics_dict['similarity'] else 0,
            'diversity': np.mean(metrics_dict['diversity']) if metrics_dict['diversity'] else 0
        }

    # Calculate average metrics for each system
    results['content'] = calc_avg_metrics(content_metrics)
    results['collaborative'] = calc_avg_metrics(collab_metrics)
    results['hybrid'] = calc_avg_metrics(hybrid_metrics)
    results['two_tower'] = calc_avg_metrics(two_tower_metrics)
    results['combined'] = calc_avg_metrics(combined_metrics)

    # Add component comparisons
    results['hybrid_component_comparison'] = hybrid_component_comparison
    results['combined_component_comparison'] = combined_component_comparison

    # Add weights used
    results['weights'] = {
        'content_weight': content_weight,
        'collaborative_weight': collaborative_weight,
        'hybrid_weight': hybrid_weight,
        'two_tower_weight': two_tower_weight
    }

    # Print formatted results
    print_formatted_results(results)

    return results


def calculate_similarity_score(recommended_games, actual_games, game_features):
    """Calculate average similarity between recommended games and actual games"""
    # Get feature vectors for actual games
    actual_game_indices = [games_df[games_df['app_id'] == game].index[0]
                           for game in actual_games
                           if game in games_df['app_id'].values]

    # Get feature vectors for recommended games
    rec_game_indices = [games_df[games_df['app_id'] == game].index[0]
                        for game in recommended_games
                        if game in games_df['app_id'].values]

    if not actual_game_indices or not rec_game_indices:
        return 0

    actual_features = game_features.iloc[actual_game_indices].values
    rec_features = game_features.iloc[rec_game_indices].values

    # Calculate similarity for each recommended game
    similarities = []
    for rec_feature in rec_features:
        reshaped_rec = rec_feature.reshape(1, -1)
        sim_scores = cosine_similarity(reshaped_rec, actual_features)[0]
        # Take the highest similarity score (closest match)
        best_match = np.max(sim_scores)
        similarities.append(best_match)

    # Return average similarity
    return np.mean(similarities) if similarities else 0


def calculate_diversity_score(games, game_features):
    """Calculate diversity as average pairwise dissimilarity between recommendations"""
    # Get feature vectors for games
    game_indices = [games_df[games_df['app_id'] == game].index[0]
                    for game in games
                    if game in games_df['app_id'].values]

    if len(game_indices) <= 1:
        return 0

    features = game_features.iloc[game_indices].values

    # Calculate pairwise similarity
    pairwise_sim = cosine_similarity(features)

    # Set diagonal elements to 0 to exclude self-similarity
    np.fill_diagonal(pairwise_sim, 0)

    # Convert similarity to dissimilarity (1 - similarity)
    pairwise_dissim = 1 - pairwise_sim

    # Return mean dissimilarity
    return np.mean(pairwise_dissim)


def print_formatted_results(results):
    """Print evaluation results in a clearly formatted way"""
    print("\n" + "=" * 80)
    print(" " * 30 + "EVALUATION RESULTS")
    print("=" * 80)

    # Define the metrics to display
    metrics = ['precision', 'recall', 'f1', 'similarity', 'diversity']

    # Function to print a section of results
    def print_section(title, section_results):
        print(f"\n{title}")
        print("-" * len(title))
        for metric in metrics:
            print(f"  {metric.capitalize():<10}: {section_results[metric]:.4f}")

    # 1. Content-Based and Collaborative Filtering (Base Models)
    print("\n" + "*" * 80)
    print(" " * 25 + "BASE RECOMMENDER MODELS")
    print("*" * 80)
    print_section("CONTENT-BASED FILTERING METRICS", results['content'])
    print_section("COLLABORATIVE FILTERING METRICS", results['collaborative'])

    # 2. Hybrid Metrics (Content + Collaborative)
    print("\n" + "*" * 80)
    print(" " * 25 + "HYBRID RECOMMENDER METRICS")
    print("*" * 80)
    print(f"\nWeights: Content ({results['weights']['content_weight']:.2f}), " +
          f"Collaborative ({results['weights']['collaborative_weight']:.2f})")
    print_section("HYBRID RECOMMENDER METRICS", results['hybrid'])

    # Component comparison in Hybrid
    comp = results['hybrid_component_comparison']
    total = comp['content_better'] + comp['collab_better'] + comp['equal']
    if total > 0:
        print("\nComponent Performance Comparison:")
        print(f"  Content-based better: {comp['content_better']} ({comp['content_better'] / total * 100:.1f}%)")
        print(f"  Collaborative better: {comp['collab_better']} ({comp['collab_better'] / total * 100:.1f}%)")
        print(f"  Equal performance:    {comp['equal']} ({comp['equal'] / total * 100:.1f}%)")

        # Determine which one performed better overall
        if comp['content_better'] > comp['collab_better']:
            print("\n  => Content-based filtering performed better in the Hybrid model")
        elif comp['collab_better'] > comp['content_better']:
            print("\n  => Collaborative filtering performed better in the Hybrid model")
        else:
            print("\n  => Both methods performed equally in the Hybrid model")

    # 3. Two-Tower Metrics
    print("\n" + "*" * 80)
    print(" " * 25 + "TWO-TOWER RECOMMENDER METRICS")
    print("*" * 80)
    print_section("TWO-TOWER RECOMMENDER METRICS", results['two_tower'])

    # 4. Combined Metrics (Hybrid + Two-Tower)
    print("\n" + "*" * 80)
    print(" " * 20 + "COMBINED RECOMMENDER METRICS (FINAL SYSTEM)")
    print("*" * 80)
    print(f"\nWeights: Hybrid ({results['weights']['hybrid_weight']:.2f}), " +
          f"Two-Tower ({results['weights']['two_tower_weight']:.2f})")
    print_section("COMBINED RECOMMENDER METRICS", results['combined'])

    # Component comparison in Combined
    comp = results['combined_component_comparison']
    total = comp['hybrid_better'] + comp['two_tower_better'] + comp['equal']
    if total > 0:
        print("\nComponent Performance Comparison:")
        print(f"  Hybrid better:     {comp['hybrid_better']} ({comp['hybrid_better'] / total * 100:.1f}%)")
        print(f"  Two-Tower better:  {comp['two_tower_better']} ({comp['two_tower_better'] / total * 100:.1f}%)")
        print(f"  Equal performance: {comp['equal']} ({comp['equal'] / total * 100:.1f}%)")

        # Determine which one performed better overall
        if comp['hybrid_better'] > comp['two_tower_better']:
            print("\n  => Hybrid recommender performed better in the Combined model")
        elif comp['two_tower_better'] > comp['hybrid_better']:
            print("\n  => Two-Tower recommender performed better in the Combined model")
        else:
            print("\n  => Both methods performed equally in the Combined model")

    # 5. Overall Model Comparison
    print("\n" + "*" * 80)
    print(" " * 30 + "OVERALL COMPARISON")
    print("*" * 80)

    # Create a table of all models for easy comparison
    models = ['content', 'collaborative', 'hybrid', 'two_tower', 'combined']
    model_names = {
        'content': 'Content-Based',
        'collaborative': 'Collaborative',
        'hybrid': 'Hybrid',
        'two_tower': 'Two-Tower',
        'combined': 'Combined'
    }

    # Find the best model for each metric
    best_models = {}
    for metric in metrics:
        best_value = max([results[model][metric] for model in models])
        best_model = [model_names[model] for model in models
                      if results[model][metric] == best_value][0]
        best_models[metric] = (best_model, best_value)

    # Print table header
    print("\nModel Comparison:")
    header = f"{'Model':<15}"
    for metric in metrics:
        header += f" | {metric.capitalize():<10}"
    print(header)
    print("-" * len(header))

    # Print each model's metrics
    for model in models:
        row = f"{model_names[model]:<15}"
        for metric in metrics:
            value = results[model][metric]
            is_best = best_models[metric][1] == value
            # Add asterisk to best value
            if is_best:
                row += f" | {value:.4f}*    "
            else:
                row += f" | {value:.4f}     "
        print(row)

    # Print summary of which model performed best overall
    print("\nBest Performer by Metric:")
    for metric in metrics:
        print(f"  {metric.capitalize():<10}: {best_models[metric][0]} ({best_models[metric][1]:.4f})")

    # Count how many metrics each model won
    model_wins = {model_name: 0 for model_name in model_names.values()}
    for metric in metrics:
        model_wins[best_models[metric][0]] += 1

    # Find overall best model
    best_model = max(model_wins.items(), key=lambda x: x[1])
    print(f"\nOverall Best Model: {best_model[0]} (best in {best_model[1]}/{len(metrics)} metrics)")

    print("\n" + "=" * 80)

# Run the comprehensive evaluation
print("\n" + "="*80)
print(" "*30 + "RUNNING COMPREHENSIVE EVALUATION")
print("="*80)

print("\nRunning comprehensive evaluation. This may take some time...")
results = evaluate_recommender_systems(
    test_data,
    game_features,
    user_game_matrix,
    content_weight=0.4,
    collaborative_weight=0.6,
    hybrid_weight=0.6,
    two_tower_weight=0.4,
    top_n=10,
    num_users=50  # You can adjust this based on computation resources
)
