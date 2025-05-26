"""
This script performs snowball crawl to iteratively collect messages and recommended channels from Telegram.

Usage:
python -m examples.snowball    --api_id <API_ID> --api_hash <API_HASH> \ 
                               --initial_csv examples/snowball_channels_sample.py \ 
                               --output_messages_file examples/results/messages.csv \
                               --output_channels_file examples/results/channels.csv \
                               --start_date 2024-08-01T00:00:00+00:00 --end_date 2024-09-01T00:00:00+00:00 \
                               --max_rounds 3                            

Modules:
    - os: Provides a way of using operating system dependent functionality.
    - telethon: Telegram client library for Python.
    - asyncio: Provides support for asynchronous programming.
    - datetime: Supplies classes for manipulating dates and times.
    - pandas: Data analysis and manipulation library.
    - rich.console: Console rendering for rich text and beautiful formatting.
    - rich.progress: Provides progress bars and spinners for console applications.
    - rich: Rich text and beautiful formatting in the terminal.
    - argparse: Parser for command-line options, arguments, and sub-commands.
    - modules.crawler: Custom module for reading channels from CSV and processing channels.    
    
Functions:
    - snowball(api_id, api_hash, initial_csv, start_date, end_date, output_file, max_rounds=3):
        Perform snowball crawl to collect messages and recommended channels iteratively.
"""

import asyncio
import csv
import datetime
import pandas as pd
from rich.console import Console
from rich.progress import Progress
from rich import print
import argparse
import os

from pytopicgram.crawler import read_channels_from_csv, process_channels

# Set up the console for rich output
console = Console()

# --------------------------------------------------------------------------------
async def snowball(api_id, api_hash, initial_csv, start_date, end_date, messages_csv, channels_csv, max_rounds=3):
    """
    Perform snowball crawl to collect messages and recommended channels iteratively.

    Parameters:
    - api_id (int): The API ID for the Telegram client.
    - api_hash (str): The API hash for the Telegram client.
    - initial_csv (str): Path to the CSV file containing the initial set of channels.
    - start_date (datetime.datetime): The start date for message retrieval.
    - end_date (datetime.datetime): The end date for message retrieval.
    - messages_csv (str): The name of the output file for saving messages.
    - channels_csv (str): The name of the output file for saving channels.
    - max_rounds (int): The maximum number of snowball rounds. Default is 3.

    Returns:
    None
    """
    temp_filename = "__temp.csv"
    temp_path = os.path.join(os.path.dirname(messages_csv), temp_filename)

    # Read and initialize the initial set of channels    
    current_channels = read_channels_from_csv(initial_csv)
    current_channels['similar_channels'] = [[] for _ in range(len(current_channels))]
    current_channels.to_csv(temp_path, index=False, mode='w', header=True)
    all_channels = pd.DataFrame()  
    visited_channels_info = pd.DataFrame()  
    round_counter = 1

    while round_counter <= max_rounds and not current_channels.empty:
        print(f"\n[bold cyan]Snowball Round {round_counter}/{max_rounds}")
        
        # Run the process_channels function to collect messages and channel info
        print(f"[cyan]Processing {len(current_channels)} channels in round {round_counter}...")
        current_channels.reset_index(drop=True, inplace=True)
        await process_channels(current_channels, start_date, end_date, api_id, api_hash, messages_csv, channels_file_name=temp_filename, append=(round_counter > 1), by_url=(round_counter == 1))
        
        # Load found channels information
        current_channels = pd.read_csv(temp_path, index_col=False)

        # Collect new channels recommended by the current set
        new_channels_list = []
        for _, row in current_channels.iterrows():
            similar_channels = row.get('similar_channels', [])
            similar_channels_list = eval(similar_channels.encode('utf-8').decode('unicode_escape'))
            for recommended_channel in similar_channels_list: 
                new_channels_list.append({
                    'id': recommended_channel['id'],
                    'channel_name': recommended_channel['title'], 
                    'url': f"https://t.me/{recommended_channel['username']}", 
                    'cluster': "not assigned",
                    'user': recommended_channel['username']
                })
        
        # Convert to DataFrame for the next round
        new_channels_df = pd.DataFrame(new_channels_list).drop_duplicates(subset='id')

        # Add current channels to the cumulative dataset
        all_channels = pd.concat([all_channels, current_channels], ignore_index=True)

        # Keep the info of visited channels
        visited_channels_info = pd.concat([visited_channels_info, current_channels], ignore_index=True)

        # Remove channels that were already processed
        new_channels_df = new_channels_df[~new_channels_df['id'].isin(all_channels['id'])]

        # Update the channels for the next round
        current_channels = new_channels_df
        round_counter += 1

        if new_channels_df.empty:
            print("[green]No new channels found. Snowball complete.")
            break

    # Save the complete list of channels processed and remove temp file
    all_channels.to_csv(channels_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    os.remove(temp_path)
    
    print(f"\n[bold green]Snowball process finished.[/bold green] Messages of processed channels saved to [bold yellow]{messages_csv}[/bold yellow].")
    print(f"\n[bold green]Visited channels info saved to [bold yellow]{channels_csv}[/bold yellow].")

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(description="Run the snowball process for Telegram channels.")
    parser.add_argument('--api_id', type=int, required=True, help='Your Telegram API ID')
    parser.add_argument('--api_hash', type=str, required=True, help='Your Telegram API hash')
    parser.add_argument('--channels_file', type=str, required=True, help='Path to your initial CSV')
    parser.add_argument('--output_channels_file', type=str, required=True, help='Desired output messages file path')
    parser.add_argument('--output_messages_file', type=str, required=True, help='Desired output channels file path')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in ISO format (e.g., 2024-08-01T00:00:00+00:00)')
    parser.add_argument('--end_date', type=str, required=True, help='End date in ISO format (e.g., 2024-09-01T00:00:00+00:00)')
    parser.add_argument('--max_rounds', type=int, default=3, help='Maximum number of snowball rounds')

    args = parser.parse_args()

    api_id = args.api_id
    api_hash = args.api_hash
    initial_csv = args.channels_file
    messages_csv = args.output_messages_file
    channels_csv = args.output_channels_file
    start_date_str = args.start_date
    end_date_str = args.end_date
    max_rounds = args.max_rounds

    # Convert date strings to datetime objects
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S%z')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M:%S%z')
    
    # Create output folders
    os.makedirs(os.path.dirname(messages_csv), exist_ok=True)
    os.makedirs(os.path.dirname(channels_csv), exist_ok=True)

    # Start the snowball process
    asyncio.run(snowball(api_id, api_hash, initial_csv, start_date, end_date, messages_csv, channels_csv, max_rounds=max_rounds))

if __name__ == "__main__":
    main()