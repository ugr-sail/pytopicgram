"""
This module is designed for preprocessing messages extracted from Telegram channels. It provides tools to extract messages from a JSON file created by crawler.py, convert them into a pandas DataFrame, and perform various preprocessing tasks.

Key features include:
- Reading and flattening JSON data into a pandas DataFrame
- Selecting specific features from the DataFrame
- Filtering messages by channel
- Processing message text to extract URLs, mentions, hashtags, and emojis
- Handling message reactions

Usage:
This module can be used as a standalone script or integrated into other scripts for further analysis or manipulation.

Functions:
- read_dataframe_from_json_file: Transforms JSON data into a flattened pandas DataFrame.
  - Parameters:
    - file_path (str): The path to the JSON file containing the messages.
    - stop_after (int, optional): The number of records to read from the JSON file. Defaults to None.

- feature_selection: Chooses specific features based on a given list.
  - Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - feature_list_file (str): The path to the CSV file containing the list of features to be selected.

- channel_selection: Filters messages by channel.
  - Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - channel_list_file (str): The path to the CSV file containing the list of channels to be selected.

- write_df_to_csv_file: Saves the processed DataFrame to a CSV file.
  - Parameters:
    - df (pd.DataFrame): The DataFrame to be written to the file.
    - output_file_path (str): The path to the output CSV file.

- process_text: Extracts various text elements from messages.
  - Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - message_column (str): The name of the column in the DataFrame that contains the text to be processed.
    - capture_urls (bool): Whether to extract URLs from the text.
    - capture_emojis (bool): Whether to extract emojis from the text.
    - capture_mentions (bool): Whether to extract mentions from the text.

- process_reactions: Converts message reactions into a JSON string.
  - Parameters:
    - reactions (dict): The reactions data as a dictionary.

Dependencies:
- argparse: Command-line argument parsing
- csv: CSV file handling
- json: JSON data parsing
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- emoji: Emoji processing
- re: Regular expressions
- urlextract: URL extraction from text
- tldextract: Top-level domain extraction
- tqdm: Progress bar for loops and iterables

This module streamlines the preprocessing of Telegram channel data, making it easier to prepare for subsequent analysis or machine learning tasks.
"""

import argparse
import csv
import json

import numpy as np
import pandas as pd

import emoji
import re
import urlextract
import tldextract

from regex_patterns import AUX_URL
from regex_patterns import MENTIONS

from tqdm import tqdm
from rich import print

# --------------------------------------------------------------------------------
def _process_reactions(reactions: dict) -> str:
    """
    Converts reactions data from a dictionary to a JSON string.

    This function processes a dictionary containing reactions data and returns a JSON string representation of the data.

    Parameters:
    - reactions (dict): A dictionary where each key is a reaction type and each value is the count of that reaction.

    Returns:
    - str: A JSON string representation of the processed reactions data.
    """
    res = {}
    for reaction in reactions:
        if 'emoticon' in reaction['reaction']:
            res[reaction['reaction']['emoticon']] = reaction['count']
        else:
            # Handle custom emoji reactions
            res[reaction['reaction']['_'] + str(reaction['reaction']['document_id'])] = reaction['count']

    return json.dumps(res)

# --------------------------------------------------------------------------------
def read_dataframe_from_json_file(json_file_path: str, stop_after: int = 0) -> pd.DataFrame:
    """
    Reads a JSON file and converts it into a pandas DataFrame with flattened nested attributes.

    Parameters:
    - json_file_path (str): The path to the JSON file to be read.
    - stop_after (int): The maximum number of records to process from the JSON file. If set to 0, all records are processed.

    Returns:
    - pd.DataFrame: A DataFrame with flattened nested attributes from the JSON file.
    """
    print(f"Reading JSON file: {json_file_path} ", end = "")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if stop_after > 0:
            data = data[:stop_after]

    for record in data:
        if record['reactions'] is not None:
            record['reactions']['results'] = _process_reactions(record['reactions']['results'])

    df = pd.json_normalize(data)
    print(f"[[green]OK[/]] ({len(df)} rows).")
    return df

# --------------------------------------------------------------------------------
def feature_selection(df: pd.DataFrame, feature_list_file: str) -> pd.DataFrame:
    """
    Select specific features from a DataFrame based on a list of feature names.

    This function reads a list of feature names from a CSV file and selects only those features from the DataFrame.
    The CSV file should contain two columns:
        - 'feature_in': The name of the feature in the input DataFrame.
        - 'feature_out': The name of the feature in the output DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.
    - feature_list_file (str): The path to the CSV file containing the list of feature names.

    Returns:
    - pd.DataFrame: A new DataFrame containing only the selected features. The names of the selected features are given by the CSV file.
    """
    print(f"Selecting features based on the list in: {feature_list_file} ", end = "")

    feature_list = pd.read_csv(feature_list_file)

    features_in = feature_list['feature_in'].tolist()
    features_out = feature_list['feature_out'].tolist()
    features_tuples = list(zip(features_in, features_out))

    selected_features_in = []
    selected_features_out = []

    for feature_in, feature_out in features_tuples:
        if feature_in in df.columns:
            selected_features_in.append(feature_in)
            selected_features_out.append(feature_out)
        else:
            print(f"[WARNING] Feature {feature_in} was not found. Results will not contain feature {feature_out}.")

    try:
        df_selected = df[selected_features_in]
        df_selected.columns = selected_features_out
        print(f"[[green]OK[/]] ({len(df_selected.columns)} columns).")
    except KeyError as e:
        df_selected = df
        print(f"[[orange]WARNING[/]] Features could not be extracted. Error: {e}")

    return df_selected

# --------------------------------------------------------------------------------
def channel_selection(df: pd.DataFrame, channel_list_file: str) -> pd.DataFrame:
    """
    Filters the DataFrame to include only messages from specific channels based on a list of channel names or IDs.

    This function reads a list of channel names or IDs from a CSV file and filters the DataFrame to include only messages from those channels.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be filtered.
    - channel_list_file (str): The path to the CSV file containing the list of channels to be included. The file must contain exactly one column named 'channel_name' (if names are provided) or 'channel_id' (if IDs are provided).

    Returns:
    - pd.DataFrame: A new DataFrame containing only messages from the specified channels.
    """
    print(f"Selecting messages from channels based on the list in: {channel_list_file} ", end = "")
    channel_list = pd.read_csv(channel_list_file)
    field = channel_list.columns[0]
    df_selected = df[df[field].isin(channel_list[field])]
    print(f"[[green]OK[/]] ({len(df_selected)} rows).")
    return df_selected


# --------------------------------------------------------------------------------
def _format_webpage_url(url: str) -> str:
    """
    Formats a URL by removing the protocol and certain trailing characters, and capitalizing the first character.

    Parameters:
    - url (str): The URL to be formatted.

    Returns:
    - str: The formatted URL with the protocol removed, the first character capitalized, and trailing characters stripped.
    """
    s = url.split("://", maxsplit=1)[-1]
    return s[0].upper() + s[1:].rstrip("\"(),.;#¡!¿?'^`")


# --------------------------------------------------------------------------------
def extract_urls(df: pd.DataFrame, message_column: str, remove_from_text: bool = True):
    """
    Extracts URLs and domains from messages, creating new columns in the DataFrame.

    This function creates two new columns in the DataFrame:
    1. 'extracted_urls': Contains the URLs extracted from the messages.
    2. 'extracted_domains': Contains the domains extracted from the URLs.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - message_column (str): The name of the column containing the message text.
    - remove_from_text (bool): If True, removes the extracted URLs from the original message text. Defaults to True.

    Returns:
    - None: The function modifies the DataFrame in-place.
    """
    print("Extracting URLs and domains from messages.")

    df.loc[:, 'extracted_urls'] = None
    df.loc[:, 'extracted_domains'] = None

    url_extractor = urlextract.URLExtract()

    # If the URL is right after a punctuation mark, it is not included
    # in the URL pattern, so we need to extract it separately
    url_pattern = re.compile(AUX_URL, re.IGNORECASE)
    space_pattern = re.compile(r"\s+")

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting URLs"):
        urls = url_pattern.findall(row[message_column])
        urls.extend(url_extractor.find_urls(row[message_column]))

        set_urls = set([_format_webpage_url(url) for url in urls])
        df.at[row.name, 'extracted_urls'] = " ".join(set_urls)

        if remove_from_text:
            for url in urls:
                df.at[row.name, message_column] = df.at[row.name, message_column].replace(url, "")

            # Remove any extra spaces that may have been left
            re.sub(space_pattern, " ", df.at[row.name, message_column]).strip()

        
        main_domains = []
        for url in set_urls:
            extracted = tldextract.extract(url).domain
            if extracted:
                main_domains.append(extracted)

        df.at[row.name, 'extracted_domains'] = main_domains

    # If message was only an url (and no description was fetched from it), remove message from df
    df.dropna(subset=[message_column], ignore_index=True, inplace=True)

    print("[[green]OK[/]] URLs have been successfully extracted.")

# --------------------------------------------------------------------------------
def extract_mentions(df: pd.DataFrame, message_column: str, remove_from_text: bool = True):
    """
    Extracts @mentions from the messages and creates a new column 'extracted_mentions' in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - message_column (str): The name of the column containing the message text.
    - remove_from_text (bool): If True, removes the extracted mentions from the original message text. Defaults to True.

    Returns:
    - None: The function modifies the DataFrame in-place by adding a new column 'extracted_mentions' and optionally removing mentions from the original messages.
    """
    regex = re.compile(MENTIONS, re.IGNORECASE)

    def _extract_mention(message):
        matches = regex.findall(message)
        if matches:
            return [match[1] or match[2] or match[3] for match in matches]

    def _remove_mention(message):
        matches = regex.findall(message)
        if matches:
            for match in matches:
                full_match = match[0]
                message = message.replace(full_match, '')
        return message.strip()

    print("Extracting mentions... ", end = "")
    df.loc[:, "extracted_mentions"] = df[message_column].apply(lambda message: _extract_mention(message)).tolist()

    if remove_from_text:
        df.loc[:, message_column] = df[message_column].apply(lambda message: _remove_mention(message)).tolist()

    print("[[green]OK[/]]")

# --------------------------------------------------------------------------------
def extract_emojis(df: pd.DataFrame, message_column: str, remove_from_text : bool = True):
    """
    Extracts emojis from the messages and creates a new column 'extracted_emojis' in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - message_column (str): The name of the column containing the message text.
    - remove_from_text (bool): If True, removes the extracted emojis from the original message text. Defaults to True.

    Returns:
    - None: The function modifies the DataFrame in-place by adding a new column 'extracted_emojis' and optionally removing emojis from the original messages.
    """
    print("Extracting emojis... ", end = "")
    emojis = []
    df.loc[:, 'extracted_emojis'] = None

    def replace_emoji_function(emoji_char, _):
        emojis.append(emoji_char)
        return ''

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting emojis"):

        text_without_emojis = emoji.replace_emoji(row[message_column], replace_emoji_function)
        df.at[idx, message_column] = text_without_emojis if text_without_emojis != '' else np.NaN

        df.at[idx, 'extracted_emojis'] = emojis

        if remove_from_text:
            df.at[idx, message_column] = text_without_emojis
            
        emojis = []

    print("[[green]OK[/]]")

# --------------------------------------------------------------------------------
def include_media_description(df: pd.DataFrame, message_column: str, desc_column: list = ["webpage_title", "webpage_description"]):
    """
    Includes media descriptions inside the text column.

    This function appends the content of specified description columns to the messages in the given message column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages.
    - message_column (str): The column where the descriptions will be added.
    - desc_column (list): The list of columns whose content will be appended to the message_column.
    """
    print("Extracting description from websites and adding it to the messages... ", end = "")
    for column in desc_column:
        if column in df.columns:
            df.loc[:, message_column] = np.where(df[column].isna(), df[message_column], df[message_column] + "\n" + df[column])
    print("[[green]OK[/]]")

# --------------------------------------------------------------------------------
def write_df_to_csv_file(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Writes the given DataFrame to a CSV file.

    This function takes a DataFrame and writes it to a CSV file at the specified path. The DataFrame index is not included in the output file, and all fields are quoted.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be written to the file.
    - output_file_path (str): The path to the output CSV file.

    Returns:
    - None
    """
    print(f"Writing CSV file: {output_file_path} ", end = "")
    df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[[green]OK[/]] ({len(df)} rows).")

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Preprocess JSON file with messages downloaded by crawler.py.")

    # Add the arguments
    parser.add_argument('-f', '--file', type=str, required=True, help="The path to the JSON file containing the messages.") 
    parser.add_argument('-o', '--output', type=str, required=True, help="The CSV output file for output data.")   
    parser.add_argument('-l', '--list-feature', type=str, required=False, help="The CSV file containing the list of features to be selected.")
    parser.add_argument('-c', '--list-channel', type=str, required=False, help="The CSV file containing the list of channels to be selected.")
    parser.add_argument('-n', '--limit', type=int, required=False, default=0, help="Limit preprocessing to the first n records.")
    parser.add_argument('-m', '--message', type=str, required=False, default="message", help="The name of the column in the DataFrame that contains the text to be processed.")
    parser.add_argument('-u', '--capture-urls', action="store_true", help="Remove URLs from the text.")
    parser.add_argument('-e', '--capture-emojis', action="store_true", help="Remove emojis from the text.")
    parser.add_argument('-d', '--add-descriptions', action="store_true", help="Add descriptions to text.")
    parser.add_argument('-cm', '--capture-mentions', action="store_true", help="Capture users or channels mentioned in the message.")

    # Parse the arguments
    args = parser.parse_args()

    input_file_path = args.file
    output_file_path = args.output
    feature_list_file = args.list_feature
    channel_list_file = args.list_channel
    limit = args.limit
    message_column = args.message
    capture_urls = args.capture_urls
    capture_emojis = args.capture_emojis
    add_descriptions = args.add_descriptions
    capture_mentions = args.capture_mentions

    # Read data
    df_input = read_dataframe_from_json_file(input_file_path, stop_after=limit)

    # Feature selection
    if feature_list_file is not None and feature_list_file != "None":
        df_input = feature_selection(df_input, feature_list_file)

    # Channel selection
    if channel_list_file is not None and channel_list_file != "None":
        df_input = channel_selection(df_input, channel_list_file)

    # Drop duplicates
    # df_selected = df_input.drop_duplicates(subset=[message_column])
    df_selected = df_input

    # Create a new column with the original message content
    df_selected[f'original_{message_column}'] = df_selected[message_column].copy()

    # Create extra column for orig_date and set to NaN if it doesn't exist
    # This is necessary, since the .json/.csv will not include this column if not found in dataset,
    # --which will happen if there are not forwarded messages in the crawled dataset--
    # and the column is needed by the metrics_calculator.py 
    if 'orig_date' not in df_selected.columns:
        df_selected['orig_date'] = pd.NaT

    # Handle URLs and emojis
    if capture_emojis:
        extract_emojis(df_selected, message_column)

    if capture_urls:
        extract_urls(df_selected, message_column)

    if capture_mentions:
        extract_mentions(df_selected, message_column)

    # Remove messages that are empty after removing elements
    df_selected.dropna(subset=[message_column], ignore_index=True, inplace=True)

    # If a message includes media (i.e. a twitter link), describe the media inside the message
    if add_descriptions:
        include_media_description(df_selected, message_column)

    # Write results
    write_df_to_csv_file(df_selected, output_file_path)

if __name__ == "__main__":
    main()
