# pytopicgram

pytopicgram is a Python library designed for extracting, processing, and topic modeling messages from Telegram channels. It provides a comprehensive pipeline for data collection, preprocessing, metrics calculation, natural language processing, and topic modeling, making it a powerful tool for researchers and analysts investigating public discourse on Telegram.

## Citation
*J. Gómez-Romero, J. Cantón-Correa, R. Pérez Mercado, F. Prados Abad, M. Molina-Solana, W. Fajardo*. **pytopicgram: A library for data extraction and topic modeling from Telegram channels**. SoftwareX 30, 102141 2025. [![DOI](https://img.shields.io/badge/doi-10.1016%2Fj.softx.2025.102141-b31b1b)](https://doi.org/10.1016/j.softx.2025.102141)




## Features

- **Fast and Flexible Message Crawling**: Efficiently connect to the Telegram API and retrieve messages from public channels using the [Telethon](https://docs.telethon.dev/en/stable/) library.
- **Extended Channel Information Retrieval**: Gather detailed information about Telegram channels, including subscriber counts, creation dates, recommended channels, and more.
- **Computation of Message Metrics**: Calculate various metrics such as virality ratios to gain insights into content reach and engagement.
- **Out-of-the-Box BERTopic Integration**: Perform topic modeling seamlessly with the [BERTopic](https://maartengr.github.io/BERTopic/index.html) algorithm, leveraging embeddings from Large Language Models (LLMs).
- **Language-Agnostic Capabilities**: Handle multiple languages, making it versatile for a wide range of linguistic contexts.
- **Data Minimization and Process Optimization**: Limit the message features stored at the beginning of the analysis, supporting data minimization and reducing dataset size.

## Installation

### Direct
To install the required dependencies and run:

```
pip install -r requirements.txt
python -m pytopicgram.main
```

### Docker (recommended)
Download the repository and then run:

```
docker build -t pytopicgram .
docker run -it pytopicgram
```

## Usage

You can run the entire pipeline through the `main.py` script. The following command initiates the process, where messages in August 2024 from `channels_sample.csv` are downloaded, preprocessed, and analyzed using topic modeling. An OpenAI key can be provided (at extra cost) to generate topic descriptions in natural language.

### Running the complete pipeline
```
python -m pytopicgram.main \
    --api_id <TELEGRAM_API_ID> --api_hash <TELEGRAM_API_HASH> \
    --start_date 2024-08-01T00:00:00+00:00 \
    --end_date 2024-09-01T00:00:00+00:00 \
    --channels_file ./pytopicgram/config/channels_sample.csv \
    --openai_key <OPENAI_KEY> \
    --description "Sample running, Aug 2024, using OpenAI API"
```

When running the `main.py` script, you can customize the behavior of the pipeline using the following parameters:

| Parameter Name                | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `--api_id`                    | Your Telegram API ID, required to access the Telegram API.                  |
| `--api_hash`                  | Your Telegram API Hash, required to access the Telegram API.                |
| `--start_date`                | The start date for collecting messages, in ISO 8601 format (e.g., `2024-08-01T00:00:00+00:00`). |
| `--end_date`                  | The end date for collecting messages, in ISO 8601 format (e.g., `2024-09-01T00:00:00+00:00`).   |
| `--channels_file`             | Path to the CSV file containing the list of channels to be processed.       |
| `--openai_key` (optional)     | Your OpenAI API key, used to generate topic descriptions in natural language. |
| `--description` (optional)    | A description of the current run, useful for logging and tracking purposes. |
| `--crawler_output_file` (optional) | Path to the output file where the crawled messages will be saved.       |
| `--preprocessor_input_file` (optional) | Path to the input file for the preprocessor.                      |
| `--preprocessor_output_file` (optional) | Path to the output file where the preprocessed messages will be saved. |
| `--preprocessor_list_feature` (optional) | List of features to be extracted during preprocessing.           |
| `--preprocessor_list_channel` (optional) | List of channels to be included during preprocessing.            |
| `--preprocessor_limit` (optional) | Limit on the number of messages to be preprocessed.                     |
| `--preprocessor_text` (optional) | Text to be used for preprocessing.                                       |
| `--remove_urls` (optional)    | Boolean flag to indicate whether URLs should be removed during preprocessing. |
| `--remove_emojis` (optional)  | Boolean flag to indicate whether emojis should be removed during preprocessing. |
| `--add_descriptions` (optional) | Boolean flag to indicate whether descriptions should be added during preprocessing. |
| `--capture_mentions` (optional) | Boolean flag to indicate whether mentions should be captured during preprocessing. |
| `--metrics_input` (optional)  | Path to the input file for metrics calculation.                             |
| `--metrics_output` (optional) | Path to the output file where the calculated metrics will be saved.         |
| `--metrics_field` (optional)  | Field to be used for metrics calculation.                                   |
| `--metrics_neighbors` (optional) | Number of neighbors to be considered for metrics calculation.            |
| `--metrics_alpha` (optional)  | Alpha value to be used for metrics calculation.                             |
| `--metrics_min_threshold` (optional) | Minimum threshold to be used for metrics calculation.                |
| `--nlp_input` (optional)      | Path to the input file for NLP processing.                                  |
| `--nlp_output` (optional)     | Path to the output file where the NLP results will be saved.                |
| `--nlp_text` (optional)       | Text to be used for NLP processing.                                         |
| `--nlp_split` (optional)      | Boolean flag to indicate whether the text should be split during NLP processing. |
| `--extractor_input_file` (optional) | Path to the input file for the topic extractor.                       |
| `--extractor_output_file` (optional) | Path to the output file where the extracted topics will be saved.    |
| `--extractor_column` (optional) | Column to be used for topic extraction.                                   |
| `--extractor_num_topics` (optional) | Number of topics to be extracted.                                     |
| `--extractor_openai` (optional) | Boolean flag to indicate whether OpenAI should be used for topic extraction. |
| `--extractor_n_docs_openai` (optional) | Number of documents to be used for OpenAI topic extraction.        |
| `--extractor_sample_ratio` (optional) | Ratio of samples to be used for topic extraction.                   |
| `--extractor_messages_used` (optional) | Number of messages to be used for topic extraction.               |
| `--viewer_model_file` (optional) | Path to the model file for the topic viewer.                             |
| `--viewer_input_file` (optional) | Path to the input file for the topic viewer.                             |
| `--viewer_training_file` (optional) | Path to the training file for the topic viewer.                       |
| `--viewer_column` (optional)  | Column to be used for the topic viewer.                                     |
| `--viewer_output` (optional)  | Path to the output file where the topic viewer results will be saved.       |
| `--viewer_num_topics` (optional) | Number of topics to be displayed in the topic viewer.                    |
| `--viewer_generate_viz` (optional) | Boolean flag to indicate whether visualizations should be generated in the topic viewer. |


### Using modules
To run any component of the library, you can use the `main` included in each one. For instance, use the following (from the project root folder) to run the `extractor`:

```bash
python -m pytopicgram.extractor \
    --file ./results/messages_nlp.csv \
    --column message_nlp \
    --output model.pkl \
    --openai_key <OPENAI_KEY> \
    --n_docs_openai 15
```

In the `examples` folder, you can find examples of running individual components of the pipeline. For instance, `snowball.py` demonstrates how to use the snowball technique to gather messages from related channels.

To run the `snowball.py` example, use the following command (from the project root folder):

```bash
python -m pytopicgram.examples.snowball \
   --api_id <TELEGRAM_API_ID> --api_hash <TELEGRAM_API_HASH> \
   --start_date 2024-08-30T00:00:00+00:00 --end_date 2024-08-31T23:59:59+00:00 \
   --channels_file ./pytopicgram/examples/snowball_channels_sample.csv \
   --output_channels_file ./results/snowball_channels.csv \
   --output_messages_file ./results/snowball_messages.json \
   --max_rounds 3
```

- `--api_id`: Your Telegram API ID, required to access the Telegram API.
- `--api_hash`: Your Telegram API Hash, required to access the Telegram API.
- `--start_date`: The start date for collecting messages, in ISO 8601 format (e.g., `2024-08-01T00:00:00+00:00`).
- `--end_date`: The end date for collecting messages, in ISO 8601 format (e.g., `2024-09-01T00:00:00+00:00`).
- `--channels_file`: Path to the CSV file containing the list of channels to be processed.
- `--output_channels_file`: Path to the output CSV file where the snowball channels will be saved.
- `--output_messages_file`: Path to the output JSON file where the collected messages will be saved.
- `--openai_key`: Your OpenAI API key, used to generate topic descriptions in natural language (optional).
- `--max_rounds`: The maximum number of rounds for the snowball process, determining how many iterations of related channel gathering will be performed.

## More information
A short video on the use of `pytopicgram`: [Watch the video on Google Drive](https://drive.google.com/file/d/1jk_b95r5dGzeNiXisPdQr4zwT1tudUIe/view?usp=share_link)

Slides: [Check the slides](https://drive.google.com/file/d/1jlfTSWeXoWEuSKQnzxAL6VZ_aNfQB-Rw/view?usp=sharing)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the UDDOT project funded by the European Media Information Fund ([EMIF](https://gulbenkian.pt/emifund/)) managed by the Calouste Gulbenkian Foundation, the XAI-DISINFODEMICS project (PLEC2021-007681) funded by MICIU/AEI/10.13039/501100011033 and by European Union NextGenerationEU/PRTR, and The Social Observatory of "la Caixa" Foundation.
