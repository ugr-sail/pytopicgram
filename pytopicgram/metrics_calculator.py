"""
This module facilitates the calculation of various metrics for a dataset. It includes functionalities to read a CSV file containing the data, compute metrics such as virality, forwards ratio, engagement ratio, and reach, and save the results to a new CSV file.

The module can be executed as an independent script or imported into other scripts for further analysis or processing.

Functions:

- virality: Computes the virality of a dataset as defined in Nobari et al. [1].
    - Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be analyzed.
        - field (str): The field to be used for calculating the virality ratio. Default is 'views'.
        - n (int): The number of neighbors to be considered. Default is 5.
        - alpha (float): The threshold used for the comparison of the virality and its neighbors. Default is 0.2.
        - min_threshold (int): The minimum threshold for the metric considered. Default is 1000.
- forwards_ratio: Computes the forwards ratio of each record in a dataset.
    - Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

- engagement_ratio: Computes the engagement ratio of each record in a dataset.
    - Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

- reach: Computes the reach of each record in a dataset.
    - Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

Dependencies:

- pandas: Library for data manipulation and analysis.
- numpy: Library for numerical computing.
- tqdm: Library for adding progress meters to loops.
- json: Module for working with JSON data.
- argparse: Parser for command-line options, arguments, and sub-commands.
"""

import pandas as pd
import numpy as np
import tqdm
import json
import argparse
from . import nlp
from rich import print

def virality(df: pd.DataFrame, field: str = 'views', n: int = 5, alpha: float = 0.2, min_threshold: int = 1000) -> pd.DataFrame:
    """
    Calculates the virality of a dataset as defined in Nobari et al. [1].

    The virality ratio is defined as the ratio between an observation and the average of its `n` neighbors.
    If the virality ratio is greater than a threshold `alpha`, the observation is considered viral.

    Parameters:

    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.
    - field (str): The field to be used for calculating the virality ratio. Default is 'views'.
    - n (int): The number of neighbors to be considered. Default is 5.
    - alpha (float): The threshold used for the comparison of the virality and its neighbors. Default is 0.2.
    - min_threshold (int): The minimum threshold for the metric considered. Default is 1000.
    - topic (str): The topic ID to be considered for the analysis. Default is None.

    Returns:

    - pd.DataFrame: A new DataFrame containing three new columns
        - `virality`: The virality of each record.
        - `virality_win_sz`: The difference in time between the first and last neighbor (self included).
        - `is_viral`: Whether the record is considered viral or not.

    [1]: Arash Dargahi Nobari, Malikeh Haj Khan Mirzaye Sarraf, Mahmood Neshati, Farnaz Erfanian Daneshvar.
    Characteristics of viral messages on Telegram; The world's largest hybrid public and private messenger.
    Expert Systems with Applications, (https://doi.org/10.1016/j.eswa.2020.114303).
    """

    print("Calculating virality... ")

    df.sort_values(by=['channel_name', 'date'], inplace=True)
    df['virality'] = None
    df['virality_win_sz'] = None

    for channel in df['channel_name'].unique():
        print(f"Calculating virality for channel: {channel}")
        df_channel = df[(df['channel_name'] == channel) & (df['orig_date'].isna())]

        if len(df_channel) < n + 1:
            print(f"Skipping channel {channel} due to insufficient data.") # @TODO: do not skip, fill with NAs
            continue

        i = 0 # index of df_channel
        for _, row in tqdm.tqdm(df_channel.iterrows(), total=len(df_channel)):

            accum = 0
            l, r = i - 1, i + 1
            l_date, r_date = row['date'], row['date']

            # get sum of field for N neighbors
            for _ in range(n):
                if l < 0:
                    accum += df_channel.iloc[r][field]
                    r_date = df_channel.iloc[r]['date']
                    r += 1
                elif r >= len(df_channel):
                    accum += df_channel.iloc[l][field]
                    l_date = df_channel.iloc[l]['date']
                    l -= 1
                elif np.abs(row['date'] - df_channel.iloc[l]['date']) < np.abs(row['date'] - df_channel.iloc[r]['date']):
                    accum += df_channel.iloc[l][field]
                    l_date = df_channel.iloc[l]['date']
                    l -= 1
                else:
                    accum += df_channel.iloc[r][field]
                    r_date = df_channel.iloc[r]['date']
                    r += 1

            # calculate the virality ratio
            df.at[row.name, 'virality'] = (row[field] / (accum / n)) - 1
            df.at[row.name, 'virality_win_sz'] = (r_date - l_date).total_seconds()

            i = i + 1
    
    df['is_viral'] = (df['virality'] >= alpha) & (df[field] >= min_threshold)

    print("[[green]OK[/]]")

    return df

# --------------------------------------------------------------------------------
def forwards_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the forwards ratio of each record in a dataset.

    The forwards ratio of a record is defined as the ratio of the number of shares to the number of views.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    - pd.DataFrame: A new DataFrame containing the forwards ratio of each record.
    """

    print("Calculating forwards ratio... ", end="")
    df['fwd_ratio'] = df['forwards'] / df['views']
    print("[[green]OK[/]]")

    return df

# --------------------------------------------------------------------------------
def engagement_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the engagement ratio of each record in a dataset.

    The engagement ratio of a record is defined as the ratio of the number of reactions to the number of views.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    - pd.DataFrame: A new DataFrame containing the engagement ratio of each record.
    """

    print("Calculating engagement ratio... ", end="")
    df['engagement_ratio'] = df['reactions'].apply(
        lambda x: sum([v for v in json.loads(x).values()]) if type(x) == str else 0
    ) / df['views']
    print("[[green]OK[/]]")

    return df

# --------------------------------------------------------------------------------
def reach(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the reach of each record in a dataset.

    The reach of a record is defined as the number of unique users who have viewed the record.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    - pd.DataFrame: A new DataFrame containing the reach of each record.
    """

    print("Calculating reach... ", end="")
    df['reach'] = df['views']
    print("[[green]OK[/]]")

    return df

# --------------------------------------------------------------------------------
def filter_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and converts the input DataFrame.

    This function performs the following operations:
    1. Filters out records with 0 views or with forward restrictions.
    2. Filters by topic if a topic is specified in the arguments.
    3. Converts the 'date' column to datetime format.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
    - pd.DataFrame: The filtered and converted DataFrame.
    """
    
    # Filter out records with 0 views or with forward restriction
    df = df[~df['no_forwards'] & (df['views'] > 0)]

    df.loc[:, 'date'] = pd.to_datetime(df['date'])

    return df

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for a dataset.')

    parser.add_argument('-i', '--input', type=str, help='Path to the input CSV file.')
    parser.add_argument('-o', '--output', type=str, help='Path to the output CSV file.')
    parser.add_argument('-f', '--field', type=str, default='views', help='Field to be used for calculating the virality ratio.')
    parser.add_argument('-n', '--neighbors', type=int, default=5, help='Number of neighbors to be considered.')
    parser.add_argument('-a', '--alpha', type=float, default=0.2, help='Threshold used for the comparison of the virality and its neighbors.')
    parser.add_argument('-m', '--min_threshold', type=int, default=1000, help='Minimum threshold for the metric considered.')
    args = parser.parse_args()

    df = nlp.read_dataframe_from_csv_file(args.input)
    df = filter_and_convert(df)
    
    df = forwards_ratio(df)
    df = engagement_ratio(df)
    df = reach(df)
    df = virality(df, field=args.field, n=args.neighbors, alpha=args.alpha, min_threshold=args.min_threshold)

    nlp.write_to_csv_file(df, args.output)

if __name__ == '__main__':
    main()