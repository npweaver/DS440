import requests
import pandas as pd
import time
import random
from tqdm import tqdm
import logging
import os


class SteamGameDetailsFetcher:
    def __init__(self, input_csv='steam_apps.csv', output_csv='steam_game_details.csv',
                 log_file='steam_game_fetch.log'):
        """
        Initialize the Steam Game Details Fetcher with aggressive data collection

        Args:
        input_csv (str): Path to the input CSV with Steam app IDs
        output_csv (str): Path to save the output CSV with game details
        log_file (str): Path to log file for tracking progress and errors
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.input_csv = input_csv
        self.output_csv = output_csv

        # Create session with retry strategy
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Tracking and resuming
        self.processed_ids = set()
        self.load_processed_ids()

    def load_processed_ids(self):
        """
        Load previously processed App IDs to allow resuming
        """
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                self.processed_ids = set(df['app_id'].tolist())
                self.logger.info(f"Loaded {len(self.processed_ids)} previously processed app IDs")
            except Exception as e:
                self.logger.warning(f"Could not load previous processed IDs: {e}")
                self.processed_ids = set()

    def fetch_game_details(self, app_id):
        """
        Comprehensive game details fetcher

        Args:
        app_id (int): Steam App ID

        Returns:
        dict: Comprehensive game details or None
        """
        try:
            # Steam Store API
            store_url = "https://store.steampowered.com/api/appdetails"
            store_params = {
                'appids': app_id,
                'l': 'english'
            }

            # Steam Spy API
            steamspy_url = "https://steamspy.com/api.php"
            steamspy_params = {
                'request': 'appdetails',
                'appid': app_id
            }

            # Fetch store details
            store_response = self.session.get(store_url, params=store_params, timeout=10)

            # Check for rate limiting
            if store_response.status_code == 429:
                self.logger.warning(f"Rate limited on App ID {app_id}. Pausing...")
                time.sleep(random.uniform(5, 15))
                return None

            store_response.raise_for_status()
            store_data = store_response.json()

            # Verify successful data retrieval
            if not (str(app_id) in store_data and store_data[str(app_id)]['success']):
                return None

            app_data = store_data[str(app_id)]['data']

            # Fetch SteamSpy details
            steamspy_details = {}
            try:
                steamspy_response = self.session.get(steamspy_url, params=steamspy_params, timeout=10)
                steamspy_response.raise_for_status()
                steamspy_details = steamspy_response.json()
            except Exception as spy_error:
                self.logger.warning(f"Could not fetch SteamSpy data for {app_id}: {spy_error}")

            # Compile comprehensive details
            game_details = {
                'app_id': app_id,
                'name': app_data.get('name', 'N/A'),
                'type': app_data.get('type', 'N/A'),
                'is_free': app_data.get('is_free', False),

                # Store API Details
                'genres': ', '.join([genre.get('description', '') for genre in app_data.get('genres', [])]),
                'categories': ', '.join([cat.get('description', '') for cat in app_data.get('categories', [])]),
                'developers': ', '.join(app_data.get('developers', ['N/A'])),
                'publishers': ', '.join(app_data.get('publishers', ['N/A'])),
                'release_date': app_data.get('release_date', {}).get('date', 'N/A'),
                'price': app_data.get('price_overview', {}).get('final_formatted', 'N/A'),

                # SteamSpy Details
                'total_owners': steamspy_details.get('owners', 'N/A'),
                'players_forever': steamspy_details.get('players_forever', 'N/A'),
                'players_2weeks': steamspy_details.get('players_2weeks', 'N/A'),
                'average_forever_playtime': steamspy_details.get('average_forever', 'N/A'),
                'average_2weeks_playtime': steamspy_details.get('average_2weeks', 'N/A')
            }

            return game_details

        except requests.RequestException as e:
            self.logger.error(f"Error fetching details for App ID {app_id}: {e}")
            return None

    def process_games(self, initial_delay=0.1, max_concurrent_fails=10):
        """
        Process all games in the input CSV

        Args:
        initial_delay (float): Initial delay between requests
        max_concurrent_fails (int): Maximum number of consecutive failed attempts before pausing
        """
        # Read input CSV
        df = pd.read_csv(self.input_csv)

        # Filtered games (skip already processed)
        games_to_process = df[~df['App ID'].isin(self.processed_ids)]

        # Progress tracking
        all_game_details = []
        concurrent_fails = 0
        current_delay = initial_delay

        # Progress bar
        progress_bar = tqdm(total=len(games_to_process), desc="Processing Games", unit="game")

        # Process games
        for index, row in games_to_process.iterrows():
            app_id = row['App ID']

            # Fetch game details
            game_details = self.fetch_game_details(app_id)

            if game_details:
                all_game_details.append(game_details)
                self.processed_ids.add(app_id)
                concurrent_fails = 0
                current_delay = initial_delay
                progress_bar.update(1)
            else:
                concurrent_fails += 1

                # Adaptive delay and pause
                if concurrent_fails >= max_concurrent_fails:
                    pause_time = random.uniform(5, 30)
                    self.logger.warning(f"Too many consecutive failures. Pausing for {pause_time:.2f} seconds")
                    time.sleep(pause_time)
                    concurrent_fails = 0
                    current_delay *= 2

                # Random small delay between requests
                time.sleep(current_delay)

            # Periodic save
            if len(all_game_details) % 100 == 0:
                self.save_progress(all_game_details)

        # Final save
        self.save_progress(all_game_details)

        progress_bar.close()
        self.logger.info(f"Completed processing. Total games processed: {len(all_game_details)}")

    def save_progress(self, game_details):
        """
        Save current progress to CSV

        Args:
        game_details (list): List of game detail dictionaries
        """
        try:
            df = pd.DataFrame(game_details)
            df.to_csv(self.output_csv, index=False)
            self.logger.info(f"Saved {len(game_details)} game details to {self.output_csv}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")


# Main execution
if __name__ == "__main__":
    fetcher = SteamGameDetailsFetcher()
    fetcher.process_games()