"""
This module facilitates the extraction of messages from Telegram channels. It offers functionalities to read a list of channels from a CSV file, connect to the Telegram API, and retrieve messages from specified channels within a given date range. The retrieved messages are then saved to a specified output file.

The module can operate as a standalone script or be imported into other scripts for further analysis or processing.

Functions:

- read_channels_from_csv: Reads channel information from a CSV file.
    - Parameters:
        - file_path (str): The path to the CSV file containing the channel information.

- process_channels: Processes each channel, retrieves messages within the specified date range, and saves them to an output file.
    - Parameters:
        - channels_list (list): A list of channels to be processed.
        - start_date (str): The start date for message retrieval in ISO format.
        - end_date (str): The end date for message retrieval in ISO format.
        - api_id (int): The Telegram API ID.
        - api_hash (str): The Telegram API hash.
        - output_file_name (str): The name of the output file where messages will be saved.
        - append (bool, optional): Whether to append to the output file if it exists. Defaults to False.

Dependencies:

- telethon: Telegram client library for Python.
- asyncio: Library to write concurrent code using the async/await syntax.
- csv: Module to read and write tabular data in CSV format.
- tqdm: Library for adding progress meters to loops.
- argparse: Parser for command-line options, arguments, and sub-commands.
- json: Module to work with JSON data.
- re: Provides support for regular expressions in Python.
"""

import os
from telethon import TelegramClient, types
from telethon.tl.functions.channels import GetFullChannelRequest, GetChannelRecommendationsRequest
import sys
import datetime
import csv
import asyncio
import argparse
import urlextract
from tqdm import tqdm
import pandas as pd
import json
import re
from rich import print
from . import regex_patterns

# The names and URLs of every channel must be present in the CSV
channel_name_csv = 'Channel'
channel_name_df  = 'channel_name'

url_name_csv     = 'URL'
url_name_df      = 'url'

user_name_csv    = 'User'
user_name_df     = 'user'

cluster_csv      = 'Tag'
cluster_df       = 'cluster'

# This dictionary represents pairs of headers found in csv files vs header names to be used
# More columns can be present in .csv file, but at least the ones listed here should appear
# The values found in this dicctionary are expected to be appended to every message
header_names = {
    channel_name_csv : channel_name_df,
    url_name_csv : url_name_df,
    user_name_csv : user_name_df,
    cluster_csv  : cluster_df
}


# --------------------------------------------------------------------------------
def read_channels_from_csv(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """
    Reads channel information from a CSV file and returns a DataFrame with channel details.

    Parameters:
        file_path (str): The path to the CSV file containing the channel information.
        delimiter (str): Character or regex pattern used to separate fields in the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the channel details with standardized column names.
    """
    print("Reading channels from CSV...")
    df = pd.read_csv(file_path, delimiter=delimiter)

    try:
        df.rename(columns=header_names, inplace=True, errors='raise')
    except KeyError as e:            
        print(f"[[orange]WARNING[/]] {e} \nExpecting {header_names} columns, \nbut found {df.columns}.")

    return df

# --------------------------------------------------------------------------------
async def process_channels(channels: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime, 
                           api_id: int, api_hash: str, 
                           output_file_name: str, channels_file_name: str = "channels_list_details.csv", 
                           append: bool = False,
                           by_url: bool = True,
                           photos: bool = True) -> None:
    """
    Asynchronously processes a list of channels, fetching messages within a specified date range and saving them to an output file. 

    Since messages are retrieved in batches, the output may include messages older than start_date.

    Parameters:
        channels (pd.DataFrame): A DataFrame where each row contains details of a channel to be processed.
        start_date (datetime.datetime): The start date from which messages should be fetched.
        end_date (datetime.datetime): The end date until which messages should be fetched.
        api_id (int): The API ID for the Telegram client.
        api_hash (str): The API hash for the Telegram client.
        output_file_name (str): The name of the JSON file where the messages should be saved.
        append (bool, optional): Determines whether to append to the output file if it already exists. Defaults to False.
        photos (bool, optional): Determines whether to download photos of not. Defaults to True.

    Returns:
        None
    """
    print(f"Starting client...", "", end="")
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()
    print(f"[[green]OK[/]]")

    filtered_messages = []
    photo_folder =  os.path.join(os.path.dirname(output_file_name), "photos")

    channels_metadata = pd.DataFrame()

    print(f"Retrieving messages from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    url_extractor = urlextract.URLExtract()
    # If the URL is right after a punctuation mark, it is not included
    # in the URL pattern, so we need to extract it separately
    url_pattern = re.compile(regex_patterns.AUX_URL, re.IGNORECASE)

    for index, channel_info in channels.iterrows():
        channel_name = channel_info[channel_name_df]
        channel_url = channel_info[url_name_df]
        channel_cluster = channel_info[cluster_df]
        
        try:
            if by_url:
                print(f"Processing channel {channel_name} at {channel_url} ({index+1}/{len(channels)})")
                channel = await client.get_entity(channel_url)
            else:
                channel_id_query = channel_info['id'] 
                print(f"Processing channel {channel_name} with id {channel_id_query} ({index+1}/{len(channels)})")
                channel = await client.get_entity(channel_id_query)
        except ValueError:
            print(f"Error: The channel {channel_name} with URL {channel_url} does not exist or is not accessible.")
            continue  # Skip to the next channel

        # Get channel metadata
        print(f"... getting channel metadata")
        channel_dict = channel.to_dict()

        # - append existing data
        channel_dict['channel_name'] = channel_name
        channel_dict['channel_url'] = channel_url
        channel_dict['cluster'] = channel_cluster

        # - profile photo
        if photos:
            os.makedirs(photo_folder, exist_ok=True)
            path = await client.download_profile_photo(channel_url, file=f"{photo_folder}/{channel_name}.jpg")
            channel_dict['photo_file'] = f"photos/{channel_name}.jpg"

        # - suscriptors
        try:
            full_info_obj = await client(GetFullChannelRequest(channel_url))
            suscriptors = full_info_obj.full_chat.participants_count
            channel_dict['suscriptors_api'] = suscriptors
        except Exception as e:
            print(f"Cannot get channel info, probably misformed or non-reachable id {channel_url}")
            continue # Skip to the next channel

        # - description
        description = full_info_obj.full_chat.about
        channel_dict['description'] = description

        urls_in_description = url_pattern.findall(description)
        urls_in_description.extend(url_extractor.find_urls(description))
        set_urls = set(urls_in_description)
        channel_dict['urls_in_description'] = " ".join(set_urls)

        # - reactions and comments enabled
        comments_enabled = full_info_obj.chats[-1].forum
        channel_dict['comments_enabled'] = comments_enabled
        reactions_enabled = full_info_obj.full_chat.available_reactions is not None
        channel_dict['reactions_enabled'] = reactions_enabled

        # - similar channels
        similar_channels_obj = await client(GetChannelRecommendationsRequest(channel_url))
        similar_channels = []
        for ch in similar_channels_obj.chats:
            similar_channel_info = {
                'username': ch.username,
                'title': ch.title,
                'id': ch.id
            }
            similar_channels.append(similar_channel_info)
        channel_dict['similar_channels'] = similar_channels

        # - pinned message
        pinned_message = await client.get_messages(channel_url, ids=types.InputMessagePinned())
        if pinned_message:
            channel_dict['pinned_message'] = pinned_message
            channel_dict['pinned_message_text'] = pinned_message.message
        
        # - add channel_dict to channels_metadata as a row
        channels_metadata = pd.concat([channels_metadata, pd.DataFrame([channel_dict])], ignore_index=True)

        # Get messages in batches
        filtered_messages_in_channel = []
        batch_size = 100
        total_fetched = 0
        date_last_fetched = end_date
        date_delta = (end_date - start_date).days
        
        with tqdm(total=date_delta, desc=f"... fetching messages by day") as pbar: 
            while date_last_fetched >= start_date:
                try:
                    messages_batch = await client.get_messages(channel, limit=batch_size, offset_date=end_date, add_offset=total_fetched)
                except Exception as e:
                    print(f"Error: Messages from channel {channel_name} older than {date_last_fetched} could not be fetched.")
                    print(f"Error code: {e}")
                    break  # Skip to the next channel

                if not messages_batch:
                    break

                for msg in messages_batch:
                    if msg.date >= start_date and msg.text is not None:
                        if len(msg.text) > 0:
                            msg_ext = msg.to_dict()
                            for col in header_names.values():
                                msg_ext[col] = channel_info[col]
                            
                            filtered_messages_in_channel.append(msg_ext)
                    else:
                        break                
                
                pbar.update((date_last_fetched-messages_batch[-1].date).days)
                
                date_last_fetched = messages_batch[-1].date
                fetched_count = len(messages_batch)                
                total_fetched += fetched_count
                
            pbar.n = pbar.total
            pbar.refresh()            
        
        print(f"... found {len(filtered_messages_in_channel)} messages")
        # if filtered_messages_in_channel:
        #    print(f"... date of the oldest message fetched: {date_last_fetched.strftime('%Y-%m-%d %H:%M:%S')}")
        filtered_messages.extend(filtered_messages_in_channel)
    
    # Write filtered_messages to JSON file
    print(f"Writing messages to JSON file {output_file_name}")
    do_append = 'a' if append else 'w'
    with open(output_file_name, do_append, newline='') as file:
        file.write(json.dumps(filtered_messages, default=str, indent=4))  

    # Write channel_dict to CSV file
    channels_csv_path = os.path.join(os.path.dirname(output_file_name), channels_file_name)
    channels_metadata.to_csv(channels_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Extended channel information saved to {channels_csv_path}")

    await client.disconnect()
    
    print(f"Process finished. Saved {len(filtered_messages)} messages in {len(channels_metadata)} channels.")

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():        
    # Create the parser
    parser = argparse.ArgumentParser(description="Download messages from channels specified in a CSV file within a date range.")

    # Add the arguments
    parser.add_argument('-f', '--file', type=str, required=True, help="The path to the CSV file containing the channels.")
    parser.add_argument('-s', '--start_date', type=str, required=False, default=datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc).isoformat(), help="The start date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to today's midnight.")
    parser.add_argument('-e', '--end_date', type=str, required=False, default=datetime.datetime.now().replace(microsecond=0, tzinfo=datetime.timezone.utc).isoformat(), help="The end date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to now.")
    parser.add_argument('-o', '--output', type=str, required=True, help="The JSON output file for the messages.")
    parser.add_argument('-a', '--append', action='store_true', required=False, help="Append to the output file if included. Defaults to False.")
    parser.add_argument('-i', '--api_id', type=int, required=True, help="The API ID for the Telegram client.")
    parser.add_argument('-p', '--api_hash', type=str, required=True, help="The API hash for the Telegram client.")
    parser.add_argument('-d', '--delimiter', type=str, required=False, default=",", help="The API hash for the Telegram client.")
    
    # Parse the arguments
    args = parser.parse_args()

    csv_file_path = args.file
    start_date_str = args.start_date
    end_date_str = args.end_date
    output_file_path = args.output
    append = args.append
    api_id = args.api_id
    api_hash = args.api_hash
    delimiter = args.delimiter

    # Get channel list from CSV file
    try:
        channels_list = read_channels_from_csv(csv_file_path, delimiter=delimiter)
        print(f"Read {len(channels_list)} channels from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} does not exist.")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error: Problem reading the file {csv_file_path} at line {e.line_num}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get dates
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S%z')
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        sys.exit(1)

    # Run crawler process
    asyncio.run(process_channels(channels_list, start_date, end_date, api_id, api_hash, output_file_path, append=append))

if __name__ == "__main__":
    main()
