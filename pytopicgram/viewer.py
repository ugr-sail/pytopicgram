"""
This module provides tools for viewing and analyzing the results of a BERTopic model. It includes functionality to load a pre-trained BERTopic model, analyze text data from CSV files, and output various insights such as topic information and document-specific details.

Functions:

- main(): The main function that parses command line arguments, loads the model and data, and performs topic analysis and reduction.


Dependencies:

- argparse: For parsing command line arguments.
- os: For file path manipulations.
- csv: For handling CSV file errors.
- pandas: For data manipulation and analysis.
- bertopic: For topic modeling with BERT.
- sys: For system-specific parameters and functions.
- rich: For enhanced console output.
"""

import argparse
import os
import csv
import pandas as pd
from bertopic import BERTopic
import sys
from rich import print

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse model to extract files with most relevant findings.")

    # Add the arguments
    parser.add_argument('-m', '--model', type=str, required=True, help="The path to the file containing the BERTopic model.")    
    parser.add_argument('-t', '--training_file', type=str, required=True, help="The path to the CSV file containing the text originally used to train the model.")    
    parser.add_argument('-f', '--file', type=str, required=True, help="The path to the CSV file containing the text to be classified by model.")        
    parser.add_argument('-c', '--column', type=str, required=True, help="Column name for the text, must be the same in both files (train and to classify).")
    parser.add_argument('-o', '--output', type=str, required=True, help="The output reduced model.")
    parser.add_argument('-n', '--num_reduced_topics', type=int, required=False, default=100, help="Number of topics to reduce to.")
    parser.add_argument('-v', '--generate_viz', action="store_true", help="Generate BERTopic visualizations for complete model and training text.")

    # Parse the arguments
    args = parser.parse_args()

    model_file_path = args.model
    training_csv_file_path = args.training_file
    classify_csv_file_path = args.file
    text_column = args.column
    output_reduced_model = args.output
    n_reduced_topics = args.num_reduced_topics
    generate_viz = args.generate_viz

    # Get the folder from output_reduced_model
    output_folder = os.path.dirname(output_reduced_model)

    # Load model
    print(f"Loading model from: {model_file_path} ", end = "")
    topic_model = BERTopic.load(model_file_path)
    num_topics = len(topic_model.get_topic_info())
    print(f"[[green]OK[/]] ({num_topics} topics)")

    # Load training messages
    print(f"Reading training CSV file: {training_csv_file_path} ", end = "")
    try:
        df_training = pd.read_csv(training_csv_file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {training_csv_file_path} does not exist.")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error: Problem reading the file {training_csv_file_path} at line {e.line_num}")
        sys.exit(1)
    training_docs = df_training[text_column].tolist()
    print(f"[[green]OK[/]] ({len(training_docs)} rows)") 

    # Load messages to classify
    print(f"Reading all messages CSV file: {classify_csv_file_path} ", end = "")
    try:
        df_classify = pd.read_csv(classify_csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {classify_csv_file_path} does not exist.")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error: Problem reading the file {classify_csv_file_path} at line {e.line_num}")
        sys.exit(1)
    classify_docs = df_classify[text_column].tolist()
    print(f"[[green]OK[/]] ({len(classify_docs)} rows)")

    # Reclassify data messages file with full model
    print("Reclassifying documents with full-size model... ")
    df_with_topics = df_classify.copy()
    df_with_topics['topics'], df_with_topics['probs'] = topic_model.transform(classify_docs)
    annotated_messages_file = os.path.join(output_folder, f"messages_annotated.csv")
    df_with_topics.to_csv(annotated_messages_file, index=False)
    print(f"[[green]OK[/]] (saved to {annotated_messages_file})")

    # Create outputs for the full model
    ## topic_info()
    print("Generating topic info for full-size model... ", end = "")
    topic_info = topic_model.get_topic_info()
    topic_info_file = os.path.join(output_folder, "topic_info.csv")
    topic_info.to_csv(topic_info_file, index=False)
    print(f"[[green]OK[/]] (saved to {topic_info_file})")

    ## document_info()
    print("Generating extended document info for full-size model... ", end="")
    document_info = pd.merge(topic_model.get_document_info(training_docs), df_training, left_index=True, right_index=True, suffixes=('_topic_model', ''))
    document_info_file = os.path.join(output_folder, "document_info_extended.csv") 
    document_info.to_csv(document_info_file, index=False)
    print(f"[[green]OK[/]] (saved to {document_info_file})")

    # Visualizations with complete model and training docs
    # https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_topics.html#visualize-topics     
    if generate_viz:     
        ## visualize_topics()
        print("Generating 2-D plot... ", end="")
        try:
            visualization = topic_model.visualize_topics()
            visualization_file = os.path.join(output_folder, "visualize_topics.html")
            visualization.write_html(visualization_file)
            print(f"[[green]OK[/]] (saved to {visualization_file})")
        except Exception as e:
            print(f"Error generating 2-D plot: {e}")

        ## visalize_heatmap()
        print("Generating heatmap... ", end="")
        try:
            heatmap = topic_model.visualize_heatmap()
            heatmap_file = os.path.join(output_folder, "visualize_heatmap.html")
            heatmap.write_html(heatmap_file)
            print(f"[[green]OK[/]] (saved to {heatmap_file})")
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    # Create smaller model by reducing the number of topics
    try:
        print("Reducing number of topics... ")
        topic_model_reduced = topic_model.reduce_topics(training_docs, nr_topics=n_reduced_topics)
        topic_model_reduced.save(output_reduced_model)
        print(f"[[green]OK[/]]")
    except Exception as e:
        print(f"[[orange]Warning[/]] Cannot reduce topic number from {num_topics} to {n_reduced_topics}: {e}")
        topic_model_reduced = None

    # Reclassify data messages file with reduced-size model
    if topic_model_reduced is not None:
        print("Reclassifying documents with reduced-size model... ")
        df_with_topics = df_classify.copy()
        df_with_topics['topics'], df_with_topics['probs'] = topic_model_reduced.transform(classify_docs)
        annotated_messages_file = os.path.join(output_folder, f"messages_annotated_{n_reduced_topics}.csv")
        df_with_topics.to_csv(annotated_messages_file, index=False)
        print(f"[[green]OK[/]] (saved to {annotated_messages_file})")

    # Create outputs for the reduced model
    if topic_model_reduced is not None:
        ## topic_info()
        print("Generating topic info for reduced-size model... ", end = "")
        topic_info = topic_model_reduced.get_topic_info()
        topic_info_file = os.path.join(output_folder, f"topic_info_{n_reduced_topics}.csv")
        topic_info.to_csv(topic_info_file, index=False)
        print(f"[[green]OK[/]] (saved to {topic_info_file})")

        ## document_info()
        print("Generating extended document info for reduced-size model... ", end = "")
        document_info = topic_model_reduced.get_document_info(training_docs)
        document_info = pd.merge(document_info, df_training, left_index=True, right_index=True, suffixes=('_topic_model', ''))
        document_info_file = os.path.join(output_folder, f"document_info_extended_{n_reduced_topics}.csv")
        document_info.to_csv(document_info_file, index=False)
        print(f"[[green]OK[/]] (saved to {document_info_file})")

if __name__ == "__main__":
    main()