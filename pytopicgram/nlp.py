"""
This module provides functions for natural language processing (NLP) tasks, including reading data from CSV files, applying NLP techniques to text data, and performing sentiment analysis. The functions can be used as standalone scripts or integrated into other Python programs for text data processing and analysis.

Functions:

- read_dataframe_from_csv_file: Reads a CSV file and converts it into a pandas DataFrame.
    - Parameters:
        - csv_file_path (str): The file path to the CSV file.

- apply_nlp: Applies NLP processing to a DataFrame containing messages.
    - Parameters:
        - df (pd.DataFrame): The DataFrame containing the messages.
        - message_column (str): The name of the column in the DataFrame that contains the text to be processed.
        - nlp_message_column (str): The name of the column to store the processed text.
        - model (str, optional): The SpaCy model to use for NLP processing. Defaults to 'es_core_news_sm'.
        - sentence_splitter (bool, optional): Whether to split the text into sentences. Defaults to False.
        - sentence_min_char (int, optional): The minimum number of characters for a sentence to be considered valid. Defaults to 50.

Dependencies:

- pandas: Provides high-performance, easy-to-use data structures and data analysis tools.
- spacy: Offers advanced natural language processing capabilities in Python.
- spacytextblob: Extends spaCy with TextBlob for sentiment analysis.
- tqdm: Adds progress meters to loops to visualize progress.
- argparse: Facilitates the creation of user-friendly command-line interfaces.

This module is designed to work with Spanish language text and uses the Spanish language model from spaCy.
"""

import argparse
import csv
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from tqdm import tqdm
from rich import print
import re
from . import regex_patterns

# --------------------------------------------------------------------------------
def read_dataframe_from_csv_file(csv_file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and converts it into a pandas DataFrame.

    Parameters:
    - csv_file_path (str): The file path to the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    print(f"Reading CSV file: {csv_file_path} ", end="")
    df = pd.read_csv(csv_file_path)
    print(f"[[green]OK[/]] ({len(df)} rows).")
    return df


def apply_nlp(df: pd.DataFrame, message_column: str, nlp_message_column: str, model: str = 'es_core_news_sm', sentence_splitter: bool = False, sentence_min_char: int = 50) -> pd.DataFrame:
    """
    Applies NLP processing to the DataFrame containing messages.

    This function processes the text of each message in the DataFrame using the SpaCy NLP pipeline, assuming the text messages are in Spanish. It performs sentiment analysis to determine polarity and subjectivity, and optionally splits the text into sentences. Each sentence (or the entire message if not split) is treated as a separate entry while preserving the original message metadata.

    The output DataFrame includes:
    - The original message data.
    - The processed text in a new column.
    - Sentiment analysis results (polarity and subjectivity).
    - A unique identifier for each original message.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the messages to be processed.
    - message_column (str): The name of the column in the DataFrame that contains the text to be processed.
    - nlp_message_column (str): The name of the column in the output DataFrame that will contain the processed text.
    - model (str): The name of the SpaCy model to use for tokenization. Defaults to 'es_core_news_sm'.
    - sentence_splitter (bool): If True, split messages into sentences. Defaults to False. Note that enumeration messages are split even when sentence_splitter is False. If True, enumeration_messages are not considered separately.
    - sentence_min_char (int): Minimum number of characters for a sentence to be included. Defaults to 50.

    Returns:
    - pd.DataFrame: A new DataFrame with the original data, processed text, and extra attributes such as sentiment analysis results and message identifiers.
    """
    # Load tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load(model)

    # Add a pipeline comment to analyze polarity (whole message)
    nlp.add_pipe("spacytextblob")

    # Add a pipeline component to split text into sentences 
    if sentence_splitter:   
        nlp.add_pipe("sentencizer") 

    # Process the message of each row and accumulate processed rows in a list before creating DataFrame
    processed_data_list = []

    msg_abs_id = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying NLP to messages..."):
        msg_text = row[message_column]
        
        # Skip to next row if len of msg_text is smaller than sentence_min_char
        if len(str(msg_text)) < sentence_min_char:
            continue

        doc = nlp(msg_text)
        
        # Sentiment analysis
        polarity = doc._.blob.polarity
        subjectivity = doc._.blob.subjectivity
        
        # Sentence splitter + enumeration finder

        if sentence_splitter:
            # - sentences are split only if sentence_splitter. This superseeds enumeration_matches.
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # - enumerations are split even if sentence_splitter is False
            enumeration_matches = re.findall(regex_patterns.NUMBERED_LIST, msg_text, re.DOTALL)
            if enumeration_matches:
                sentences = enumeration_matches
            else:
                sentences = [msg_text]

        # Create new rows, skipping sentences with less than sentence_min_char characters
        for sentence in sentences:
            
            if len(sentence) >= sentence_min_char:
                new_row = row.copy()
                new_row[nlp_message_column] = sentence
                new_row['polarity'] = polarity
                new_row['subjectivity'] = subjectivity
                new_row['msg_abs_id'] = msg_abs_id
                processed_data_list.append(new_row)

        # Increment absolute message id
        msg_abs_id += 1

    print("Building the processed data frame...")

    # Create a DataFrame from the processed rows list
    processed_df = pd.DataFrame(processed_data_list)

    # Remove duplicates
    # print("Removing duplicated sentences.") if sentence_splitter else print("Removing duplicated messages.")
    # processed_df = processed_df.drop_duplicates(subset=[nlp_message_column])

    print("[[green]OK[/]] NLP has been successfully applied.")

    return processed_df

# --------------------------------------------------------------------------------
def write_to_csv_file(df: pd.DataFrame, csv_file_path: str):
    """
    Writes the provided DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be written to the CSV file.
    - csv_file_path (str): The file path where the CSV file will be written.
    """
    print(f"Writing data frame to CSV file: {csv_file_path} ", end = "")
    df.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[[green]OK[/]] ({len(df)} rows).")

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Preprocess the messages in the CSV file provided by preprocessor.py.")

    # Add the arguments
    parser.add_argument('-f', '--file', type=str, required=True, help="The path to the CSV file containing the messages.") 
    parser.add_argument('-o', '--output', type=str, required=True, help="The CSV output file for output data.")   
    parser.add_argument('-t', '--message', type=str, required=False, default="message", help="The name of the column in the DataFrame that contains the text to be processed, i.e. the message.")
    parser.add_argument('-s', '--split',  action="store_true", help="Split messages into sentences. Defaults to false.")

    # Parse the arguments
    args = parser.parse_args()

    input_file_path = args.file
    output_file_path = args.output
    text_column = args.message
    split = args.split

    # Read data
    df_input = read_dataframe_from_csv_file(input_file_path)

    # Apply NLP
    df_output = apply_nlp(df_input, text_column, f'{text_column}_nlp', sentence_splitter=split)

    # Write results
    write_to_csv_file(df_output, output_file_path)

if __name__ == "__main__":
    main()


