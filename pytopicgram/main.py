"""
This module is designed to automatically execute the crawler -> preprocessor -> metrics -> extractor -> viewer pipeline

.. code-block:: bash

    python executor.py -i <TELEGRAM_API_ID> -p <TELEGRAM_API_HASH>

"""

import argparse
import datetime
import subprocess
import os
import shutil
import json
import time
from rich import print

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def main():
    """
    Main function to execute the pipeline stages: crawler, preprocessor, metrics, extractor, and viewer.

    This function sets up an argument parser to handle command-line arguments for each stage of the pipeline.
    It orchestrates the execution of the stages in sequence, ensuring that the output of one stage is used as
    the input for the next.

    Command-line Arguments:

    - **-d, --description**: 
      A sentence that describes this execution, for instance, the reason why it was executed.

    - **-ch, --channels_file**: 
      The path to the CSV file containing the channel list.

    - **-s, --start_date**: 
      The start date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to today's midnight.

    - **-e, --end_date**: 
      The end date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to now.

    - **-co, --crawler_output**: 
      The JSON output file for the messages obtained with the crawler. Defaults to results/datetime/channel_messages.json.

    - **-a, --append**: 
      Append to the output file if included. Defaults to False.

    - **-i, --api_id**: 
      The API ID for the Telegram client.

    - **-p, --api_hash**: 
      The API hash for the Telegram client.

    - **-de, --delimiter**: 
      Delimiter for channels CSV file.

    - **-pi, --preprocessor_input**: 
      The JSON input file for the preprocessor containing the messages obtained with the crawler. Defaults to results/datetime/channel_messages.json.

    - **-po, --preprocessor_output**: 
      The CSV output file for the preprocessor output data. Defaults to results/datetime/messages_preprocessed.csv.

    - **-l, --list-feature**: 
      The CSV file containing the list of features to be selected.

    - **-c, --list-channel**: 
      The CSV file containing the list of channels to be selected.

    - **-nl, --limit**: 
      Limit preprocessing to the first n records. Defaults to 0, which results in all records.

    - **-pt, --preprocessor_text**: 
      The name of the column in the DataFrame that contains the text to be processed.

    - **-cu, --capture_urls**: 
      Capture URLs from the text.

    - **-ce, --capture_emojis**: 
      Capture emojis from the text.

    - **-ad, --add_descriptions**: 
      Add media descriptions to text when links are present.

    - **-cm, --capture_mentions**: 
      Capture users or channels mentioned in the message.

    - **-mi, --metrics_input_file_path**: 
      Path to the input CSV file for the metrics calculation.

    - **-mo, --metrics_output_file_path**: 
      Path to the output CSV file for the metrics calculation.

    - **-mf, --metrics_field**: 
      Field to be used for calculating the virality ratio.

    - **-mn, --metrics_neighbors**: 
      Number of neighbors to be considered. Defaults to 5.

    - **-ma, --metrics_alpha**: 
      Threshold used for the comparison of the virality and its neighbors. Defaults to 0.2.

    - **-mm, --metrics_min_threshold**: 
      Minimum threshold for the metric considered. Defaults to 1000.

    - **-ni, --nlp_input**: 
      The path to the CSV file containing the preprocessed messages with metrics. Defaults to results/datetime/messages_preprocessed.csv.

    - **-no, --nlp_output**: 
      The CSV output file for the nlp output data. Defaults to results/datetime/messages_nlp.csv.

    - **-nt, --nlp_text**: 
      The name of the column in the DataFrame that contains the text to be processed. Defaults to message.

    - **-ns, --nlp_split**: 
      Split messages into sentences.
    """

    execution_datetime = datetime.datetime.now()
    results_folder = "results/" + execution_datetime.strftime('%Y-%m-%d_%H_%M_%S') + "/"

    os.makedirs(results_folder)
    print("Results will be saved to folder " + results_folder)

    # Create the parser
    parser = argparse.ArgumentParser(description="Download messages from channels specified in a CSV file within a date range.")

    # Add the executor arguments
    parser.add_argument('-d', '--description',          type=str, required=False, help="A sentence that describes this execution, for instance the reason why it was executed.")

    # Add the crawler arguments
    parser.add_argument('-ch', '--channels_file',       type=str, required=False,  default="config/channels.csv", help="The path to the CSV file containing the channel list.")
    parser.add_argument('-s', '--start_date',           type=str, required=False, default=datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc).isoformat(), help="The start date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to today's midnight.")
    parser.add_argument('-e', '--end_date',             type=str, required=False, default=datetime.datetime.now().replace(microsecond=0, tzinfo=datetime.timezone.utc).isoformat(), help="The end date in the format 'YYYY-MM-DDTHH:MM:SS+00:00'. Defaults to now.")
    parser.add_argument('-co', '--crawler_output',      type=str, required=False, default=results_folder + "channel_messages.json", help="The JSON output file for the messages obtained with the crawler. Defaults to results/datetime/channel_messages.json")
    parser.add_argument('-a', '--append', action='store_true',    required=False, help="Append to the output file if included. Defaults to False.")
    parser.add_argument('-i', '--api_id',               type=int, required=True, help="The API ID for the Telegram client.")
    parser.add_argument('-p', '--api_hash',             type=str, required=True, help="The API hash for the Telegram client.")
    parser.add_argument('-de', '--delimiter',           type=str, required=False, default=",", help="Delimiter for channels CSV file.")
    
    # Add the preprocessor arguments
    parser.add_argument('-pi', '--preprocessor_input',  type=str, required=False, default=results_folder + "channel_messages.json", help="The JSON input file for the preprocessor containing the messages obtained with the crawler. Defaults to results/datetime/channel_messages.json")
    parser.add_argument('-po', '--preprocessor_output', type=str, required=False, default=results_folder + "messages_preprocessed.csv", help="The CSV output file for the preprocessor output data. Defaults to results/datetime/messages_preprocessed.csv")
    parser.add_argument('-l', '--list-feature',         type=str, required=False, default="config/list_features.csv", help="The CSV file containing the list of features to be selected.")
    parser.add_argument('-c', '--list-channel',         type=str, required=False, help="The CSV file containing the list of channels to be selected.")
    parser.add_argument('-nl', '--limit',               type=int, required=False, default=0, help="Limit preprocessing to the first n records. Defaults to 0, which results in all records.")
    parser.add_argument('-pt', '--preprocessor_text',   type=str, required=False, default="message", help="The name of the column in the DataFrame that contains the text to be processed.")
    parser.add_argument('-cu', '--capture_urls',      action="store_true", help="Capture URLs from the text.")
    parser.add_argument('-ce', '--capture_emojis',    action="store_true", help="Capture emojis from the text.")
    parser.add_argument('-ad', '--add_descriptions', action="store_true", help="Add media descriptions to text when links are present.")
    parser.add_argument('-cm', '--capture_mentions', action="store_true", help="Capture users or channels mentioned in the message.")

    # Add metric calculation parameters
    parser.add_argument('-mi', '--metrics_input_file_path', type=str, required=False, default=results_folder + "messages_preprocessed.csv", help='Path to the input CSV file for the metrics calculation.')
    parser.add_argument('-mo', '--metrics_output_file_path', type=str, required=False, default=results_folder + "messages_metrics.csv", help='Path to the output CSV file for the metrics calculation.')
    parser.add_argument('-mf', '--metrics_field', type=str, default='views', help='Field to be used for calculating the virality ratio.')
    parser.add_argument('-mn', '--metrics_neighbors', type=int, default=5, help='Number of neighbors to be considered. Defaults to 5.')
    parser.add_argument('-ma', '--metrics_alpha', type=float, default=0.2, help='Threshold used for the comparison of the virality and its neighbors. Defaults to 0.2.')
    parser.add_argument('-mm', '--metrics_min_threshold', type=int, default=1000, help='Minimum threshold for the metric considered. Defaults to 1000.')

    # Add the nlp arguments
    parser.add_argument('-ni', '--nlp_input',           type=str, required=False, default=results_folder + "messages_metrics.csv", help="The path to the CSV file containing the preprocessed messages with metrics. Defaults to results/datetime/messages_preprocessed.csv")
    parser.add_argument('-no', '--nlp_output',          type=str, required=False, default=results_folder + "messages_nlp.csv", help="The CSV output file for the nlp output data. Defaults to results/datetime/messages_nlp.csv")
    parser.add_argument('-nt', '--nlp_text',            type=str, required=False, default="message", help="The name of the column in the DataFrame that contains the text to be processed. Defaults to message.")
    parser.add_argument('-ns', '--nlp_split',           action="store_true", help="Split messages into sentences.")

    # Add the extractor arguments
    parser.add_argument('-ei', '--extractor_input',     type=str, required=False, default=results_folder + "messages_nlp.csv", help="The path to the CSV file containing the messages. Defaults to results/datetime/messages_nlp.csv")
    parser.add_argument('-eo', '--extractor_output',    type=str, required=False, default=results_folder + "model.pkl", help="The output file for the model. Defaults to results/datetime/model.pkl.")
    parser.add_argument('-ec', '--extractor_column',    type=str, required=False, default="message_nlp", help="Column name for the text. Defaults to message_processed.")
    parser.add_argument('-en', '--num_topics',          type=int, required=False, default=1000, help="Number of topics to generate. Defaults to 1000.")
    parser.add_argument('-ek', '--openai_key', type=str, required=False, help="If an OpenAI key is provided, a description with OpenAI is generated for the topics.")
    parser.add_argument('-ed', '--n_docs_openai', type=int, required=False, default=10, help="If an OpenAI key is provided, the number of documents used to find the description. Defaults to 10.")
    parser.add_argument('-er', '--extractor_sample_ratio', type=float, required=False, help="Builds the topic model only with a sample of messages with size sample_ratio * number_of_messages.")
    parser.add_argument('-em', '--extractor_messages_used', type=str, required=False, default=results_folder + "messages_used_for_model.csv", help="The path of the CSV where the messages used to train the model will be stored.")

    # Add the viewer arguments
    parser.add_argument('-m', '--model',                type=str, required=False, default=results_folder + "model.pkl", help="The path to the file containing the BERTopic model. Defaults to results/datetime/model.pkl")
    parser.add_argument('-vi', '--viewer_input',        type=str, required=False, default=results_folder + "messages_nlp.csv", help="The path to the CSV file containing the text to be used by the viewer. Defaults to results/datetime/messages_nlp.csv")
    parser.add_argument('-vt', '--viewer_training',     type=str, required=False, default=results_folder + "messages_used_for_model.csv", help="The path to the CSV file containing the messages used in training. Defaults to results/datetime/messages_used_for_model.csv")
    parser.add_argument('-vc', '--viewer_column',       type=str, required=False, default="message_nlp", help="Column name for the text. Defaults to message_nlp")
    parser.add_argument('-vo', '--viewer_output',       type=str, required=False, default=results_folder + "model_reduced.pkl", help="The output file for the model information (.zip).")
    parser.add_argument('-nr', '--num_reduced_topics',  type=int, required=False, default=100, help="Number of topics to reduce to. Defaults to 100.")
    parser.add_argument('-vv', '--generate_viz',        action="store_true", help="Generate BERTopic visualizations for complete model.")

    # Add the profiler arguments
    parser.add_argument('-pr', '--profiling', action='store_true', help="Enable profiling to make run_process not silent.")

    # Parse the arguments
    args = parser.parse_args()

    # Executor arguments
    executor_description = args.description

    # Crawler arguments
    csv_channels_list = args.channels_file
    start_date_str = args.start_date
    end_date_str = args.end_date
    crawler_output_file_path = args.crawler_output
    append = args.append
    api_id = args.api_id
    api_hash = args.api_hash
    delimiter = args.delimiter

    # Preprocessor arguments 
    preprocessor_input_file_path = args.preprocessor_input
    preprocessor_output_file_path = args.preprocessor_output
    list_feature = args.list_feature
    list_channel = args.list_channel
    limit = args.limit
    preprocessor_text = args.preprocessor_text
    capture_urls = args.capture_urls
    capture_emojis = args.capture_emojis
    add_descriptions = args.add_descriptions
    capture_mentions = args.capture_mentions

    # Metric calculation arguments
    metrics_input_file_path = args.metrics_input_file_path
    metrics_output_file_path = args.metrics_output_file_path
    metrics_field = args.metrics_field
    metrics_neighbors = args.metrics_neighbors
    metrics_alpha = args.metrics_alpha
    metrics_min_threshold = args.metrics_min_threshold

    # NLP arguments
    nlp_input_file_path = args.nlp_input
    nlp_output_file_path = args.nlp_output
    nlp_text = args.nlp_text
    nlp_split = args.nlp_split

    # Extractor arguments
    extractor_input_file_path = args.extractor_input
    extractor_output_file_path = args.extractor_output
    extractor_column = args.extractor_column
    extractor_num_topics = args.num_topics
    openai_key = args.openai_key
    n_docs_openai = args.n_docs_openai
    extractor_sample_ratio = args.extractor_sample_ratio
    extractor_messages_used = args.extractor_messages_used

    # Viewer arguments
    viewer_model_file_path = args.model
    viewer_input_file_path = args.viewer_input
    viewer_training_file_path = args.viewer_training 
    viewer_column = args.viewer_column
    viewer_output = args.viewer_output
    n_reduced_topics = args.num_reduced_topics
    viewer_generate_viz = args.generate_viz

    # Profiler arguments
    profiler = args.profiling
    if profiler:        
        print("[[green]INFO[/]] Time ticking is enabled.")

    if crawler_output_file_path != preprocessor_input_file_path:
        print("[WARNING] The output of the executed crawler will NOT be passed to the preprocessor. Make sure this is expected")

    if preprocessor_output_file_path != metrics_input_file_path:
        print("[WARNING] The output of the executed preprocessor will NOT be passed to the metrics calculation. Make sure this is expected")

    if metrics_output_file_path != nlp_input_file_path:
        print("[WARNING] The output of the metrics calculator will NOT be passed to the NLP. Make sure this is expected")

    if nlp_output_file_path != extractor_input_file_path:
        print("[WARNING] The output of the executed preprocessor will NOT be passed to the extractor. Make sure this is expected")

    if extractor_output_file_path != viewer_model_file_path:
        print("[WARNING] The output of the executed extractor will NOT be passed to the viewer. Make sure this is expected")

    if nlp_output_file_path != viewer_input_file_path:
        print("[WARNING] The output of the executed NLP will NOT be passed to the viewer. Make sure this is expected")

    scripts_folder = "./"

    crawler         = scripts_folder + "crawler.py"
    preprocessor    = scripts_folder + "preprocessor.py"
    metrics         = scripts_folder + "metrics_calculator.py"
    nlp             = scripts_folder + "nlp.py"
    extractor       = scripts_folder + "extractor.py"
    viewer          = scripts_folder + "viewer.py"

    def run_subprocess(command):
        start_time = time.time()
        subprocess.run(command)
        end_time = time.time()
        if profiler:
            print(f"Execution time for {' '.join(command)}: {end_time - start_time:.2f} seconds")

    if api_id and api_hash:
        # Execute the crawler
        args = [
            "-f", csv_channels_list,
            "-s", start_date_str,
            "-e", end_date_str,
            "-o", crawler_output_file_path,
            "-i", str(api_id),
            "-p", api_hash,
            "-d", delimiter
        ]
        if append:
            args.append("-a")

        run_subprocess(["python", crawler] + args)
    else:
        print("Missing api_id and/or api_hash: crawler cannot be executed.")

    if os.path.exists(preprocessor_input_file_path):
        # Execute the preprocessor
        args = [
            "-f", preprocessor_input_file_path,
            "-o", preprocessor_output_file_path,
            "-l", str(list_feature),
            "-c", str(list_channel),
            "-n", str(limit),
            "-m", preprocessor_text
        ]
        if capture_urls:
            args.append("-u")
        if capture_emojis:
            args.append("-e")
        if add_descriptions:
            args.append("-d")
        if capture_mentions:
            args.append("-cm")
        run_subprocess(["python", preprocessor] + args)
    else:
        print(f"{preprocessor_input_file_path} does not exist. preprocessor cannot be executed.")

    if os.path.exists(metrics_input_file_path):
        # Execute the metrics calculator
        args = [
            "-i", metrics_input_file_path,
            "-o", metrics_output_file_path,
            "-f", metrics_field,
            "-n", str(metrics_neighbors),
            "-a", str(metrics_alpha),
            "-m", str(metrics_min_threshold)
        ]
        run_subprocess(["python", metrics] + args)
    else:
        print(f"{metrics_input_file_path} does not exist. metrics calculation cannot be executed.")

    if os.path.exists(nlp_input_file_path):
        # Execute NLP
        args = [
            "-f", nlp_input_file_path,
            "-o", nlp_output_file_path,
            "-t", nlp_text
        ]
        if nlp_split:
            args.append("-s")
        run_subprocess(["python", nlp] + args)
    else:
        print(f"{nlp_input_file_path} does not exist. nlp cannot be executed.")

    if os.path.exists(extractor_input_file_path):
        # Execute the extractor
        args = [
            "-f", extractor_input_file_path,
            "-o", extractor_output_file_path,
            "-c", extractor_column,
            "-n", str(extractor_num_topics),
            "-nd", str(n_docs_openai),
            "-m", extractor_messages_used
        ]
        if extractor_sample_ratio:
            args.append("-r")
            args.append(str(extractor_sample_ratio))
        if openai_key:
            args.append("-k")
            args.append(openai_key)
        run_subprocess(["python", extractor] + args)
    else:
        print(f"{extractor_input_file_path} does not exist. extractor cannot be executed.")

    if os.path.exists(viewer_model_file_path) and os.path.exists(viewer_input_file_path) and os.path.exists(viewer_training_file_path):
        # Execute the viewer
        args = [
            "-m", viewer_model_file_path,
            "-f", viewer_input_file_path,
            "-c", viewer_column,
            "-o", viewer_output,
            "-n", str(n_reduced_topics),
            "-t", viewer_training_file_path
        ]
        if viewer_generate_viz:
            args.append("-v")
        run_subprocess(["python", viewer] + args)
    else:
        print(f"{viewer_model_file_path} and/or {viewer_input_file_path} and/or {viewer_training_file_path} do not exist. viewer cannot be executed.")

    # In order to be able to reproduce the experiment, save some metadata
    shutil.copy2(csv_channels_list, results_folder + "channels_list_used.csv")  # copy2 tries to preserve file's metadata

    info = {
        # "username": os.getlogin(),
        "description": executor_description,

        "execution_start": execution_datetime.strftime('%Y-%m-%d_%H_%M_%S'),
        "execution_end": datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),

        "messages_since": start_date_str,
        "messages_until": end_date_str,

        "crawler_input_file": csv_channels_list,
        "crawler_output_file": crawler_output_file_path,

        "preprocessor_input_file": preprocessor_input_file_path,
        "preprocessor_output_file": preprocessor_output_file_path,
        "preprocessor_list_feature": list_feature,
        "preprocessor_list_channel": list_channel,
        "preprocessor_limit": limit,
        "preprocessor_text": preprocessor_text,
        "remove_urls": capture_urls,
        "remove_emojis": capture_emojis,
        "add_descriptions": add_descriptions,
        "capture_mentions": capture_mentions,

        "metrics_input": metrics_input_file_path,
        "metrics_output": metrics_output_file_path,
        "metrics_field": metrics_field,
        "metrics_neighbors": metrics_neighbors,
        "metrics_alpha": metrics_alpha,
        "metrics_min_threshold": metrics_min_threshold,

        "nlp_input": nlp_input_file_path,
        "nlp_output": nlp_output_file_path,
        "nlp_text": nlp_text,
        "nlp_split": nlp_split,

        "extractor_input_file": extractor_input_file_path,
        "extractor_output_file": extractor_output_file_path,
        "extractor_column": extractor_column,
        "extractor_num_topics": extractor_num_topics,
        "extractor_openai": "yes" if openai_key else "no",
        "extractor_n_docs_openai": n_docs_openai,
        "extractor_sample_ratio": extractor_sample_ratio,
        "extractor_messages_used": extractor_messages_used,

        "viewer_model_file": viewer_model_file_path,
        "viewer_input_file": viewer_input_file_path,
        "viewer_training_file": viewer_training_file_path,
        "viewer_column": viewer_column,
        "viewer_output": viewer_output,
        "viewer_num_topics": n_reduced_topics,
        "viewer_generate_viz": viewer_generate_viz
    }

    log_file = results_folder + "info.json"
    with open(log_file, 'w') as file:
        file.write(json.dumps(info))

    print(f"Log stored in {log_file}")
    print(f"[bold green]Process finished[/bold green]")

if __name__ == "__main__":
    main()
