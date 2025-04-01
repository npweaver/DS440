import requests
import pandas as pd
import time
import os
from pathlib import Path


def get_app_details(app_id):
    """
    Get detailed information about a Steam app using its app_id.
    Uses the Steam Store API.
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data[str(app_id)]['success']:
            return data[str(app_id)]['data']
        else:
            return None
    except Exception as e:
        print(f"Error fetching details for app {app_id}: {e}")
        return None


def get_app_stats(app_id):
    """
    Get player stats for a Steam app using its app_id.
    Uses the Steam API for current player count.
    """
    url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={app_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['response']['result'] == 1:
            return data['response']['player_count']
        else:
            return None
    except Exception as e:
        print(f"Error fetching stats for app {app_id}: {e}")
        return None


def extract_app_info(app_data):
    """
    Extract relevant information from the app data.
    """
    if not app_data:
        return {}

    info = {
        'name': app_data.get('name', ''),
        'type': app_data.get('type', ''),
        'required_age': app_data.get('required_age', 0),
        'is_free': app_data.get('is_free', False),
        'short_description': app_data.get('short_description', ''),
        'supported_languages': app_data.get('supported_languages', ''),
        'header_image': app_data.get('header_image', ''),
        'website': app_data.get('website', ''),
        'release_date': app_data.get('release_date', {}).get('date', '') if app_data.get('release_date') else '',
        'coming_soon': app_data.get('release_date', {}).get('coming_soon', False) if app_data.get(
            'release_date') else False,
        'price': app_data.get('price_overview', {}).get('final_formatted', 'Free') if app_data.get(
            'price_overview') else 'Free',
        'metacritic_score': app_data.get('metacritic', {}).get('score', None) if app_data.get('metacritic') else None,
        'recommendations': app_data.get('recommendations', {}).get('total', 0) if app_data.get(
            'recommendations') else 0,
        'achievements': app_data.get('achievements', {}).get('total', 0) if app_data.get('achievements') else 0,
        'platforms': ', '.join([p for p, support in app_data.get('platforms', {}).items() if support]),
    }

    # Categories (like Single-player, Multi-player, etc.)
    if app_data.get('categories'):
        info['categories'] = ', '.join([cat['description'] for cat in app_data['categories']])

    # Genres
    if app_data.get('genres'):
        info['genres'] = ', '.join([genre['description'] for genre in app_data['genres']])

    # Tags (from Steam store)
    if app_data.get('tags'):
        info['tags'] = ', '.join([tag for tag in app_data['tags']])

    # Developers
    if app_data.get('developers'):
        info['developers'] = ', '.join(app_data['developers'])

    # Publishers
    if app_data.get('publishers'):
        info['publishers'] = ', '.join(app_data['publishers'])

    return info


def save_checkpoint(app_details_list):
    """
    Save a checkpoint CSV file with the data processed so far.
    Always overwrites the same file.
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Create a DataFrame with the collected information
    result_df = pd.DataFrame(app_details_list)

    # Reorder columns to put App ID and App Name first
    columns = result_df.columns.tolist()
    if 'App ID' in columns and 'App Name' in columns:
        columns.remove('App ID')
        columns.remove('App Name')
        columns = ['App ID', 'App Name'] + columns
        result_df = result_df[columns]

    # Save to a single checkpoint CSV file (overwriting the same file)
    output_path = checkpoint_dir / "steam_apps_details_checkpoint.csv"
    result_df.to_csv(output_path, index=False)

    print(f"Checkpoint saved: {output_path} ({len(app_details_list)} entries)")


def load_checkpoint():
    """
    Load the checkpoint file if it exists.
    Returns a list of processed app IDs and the app details data.
    """
    checkpoint_dir = Path("checkpoints")
    checkpoint_file = checkpoint_dir / "steam_apps_details_checkpoint.csv"

    if not checkpoint_dir.exists() or not checkpoint_file.exists():
        return set(), []

    try:
        df = pd.read_csv(checkpoint_file)
        processed_app_ids = set(df['App ID'].tolist())
        app_details_list = df.to_dict('records')
        print(f"Loaded {len(app_details_list)} records from {checkpoint_file}")
        return processed_app_ids, app_details_list
    except Exception as e:
        print(f"Error reading checkpoint file {checkpoint_file}: {e}")
        return set(), []


def main():
    # Load the CSV file containing app IDs
    input_path = Path('steam_apps.csv')
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure the CSV has an 'App ID' column
    if 'App ID' not in df.columns:
        print("Error: CSV file must contain an 'App ID' column.")
        return

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Load existing checkpoint data
    processed_app_ids, app_details_list = load_checkpoint()
    print(f"Found {len(processed_app_ids)} already processed app IDs.")

    # Filter out already processed app IDs
    df_to_process = df[~df['App ID'].isin(processed_app_ids)]

    # Process each app ID
    total_to_process = len(df_to_process)
    print(f"Processing {total_to_process} remaining app IDs.")

    for i, (_, row) in enumerate(df_to_process.iterrows()):
        app_id = row['App ID']
        app_name = row['App Name'] if 'App Name' in df.columns else 'Unknown'

        print(f"Processing app {i + 1}/{total_to_process}: {app_name} (ID: {app_id})")

        # Get app details from Steam store
        app_data = get_app_details(app_id)

        # Extract relevant information
        app_info = extract_app_info(app_data)

        # Get current player count
        player_count = get_app_stats(app_id)
        if player_count is not None:
            app_info['current_players'] = player_count

        # Add app ID and name to the information
        app_info['App ID'] = app_id
        app_info['App Name'] = app_name

        # Add to the list
        app_details_list.append(app_info)

        # Save checkpoint every 100 apps or at the last app
        if (i + 1) % 100 == 0 or i == total_to_process - 1:
            save_checkpoint(app_details_list)

        # Sleep to avoid hitting rate limits
        time.sleep(1)

    # Save final output CSV
    if app_details_list:
        result_df = pd.DataFrame(app_details_list)

        # Reorder columns to put App ID and App Name first
        columns = result_df.columns.tolist()
        if 'App ID' in columns and 'App Name' in columns:
            columns.remove('App ID')
            columns.remove('App Name')
            columns = ['App ID', 'App Name'] + columns
            result_df = result_df[columns]

        # Save to the final CSV file
        output_path = Path('steam_apps_details.csv')
        result_df.to_csv(output_path, index=False)

        print(f"Completed! Final data saved to {output_path}")
    else:
        print("No data was processed.")


if __name__ == "__main__":
    main()
