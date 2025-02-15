pytopicgram
===========

pytopicgram is a Python library designed for extracting, processing, and
topic modeling messages from Telegram channels. It provides a
comprehensive pipeline for data collection, preprocessing, metrics
calculation, natural language processing, and topic modeling, making it
a powerful tool for researchers and analysts investigating public
discourse on Telegram.

Citation
--------

*J. Gómez-Romero, J. Cantón-Correa, R. Pérez Mercado, F. Prados Abad, M.
Molina-Solana, W. Fajardo*. **pytopicgram: A library for data extraction
and topic modeling from Telegram channels**. September 2024.
|DOI:10.5281/zenodo.13863008|

Features
--------

-  **Fast and Flexible Message Crawling**: Efficiently connect to the
   Telegram API and retrieve messages from public channels using the
   `Telethon <https://docs.telethon.dev/en/stable/>`__ library.
-  **Extended Channel Information Retrieval**: Gather detailed
   information about Telegram channels, including subscriber counts,
   creation dates, recommended channels, and more.
-  **Computation of Message Metrics**: Calculate various metrics such as
   virality ratios to gain insights into content reach and engagement.
-  **Out-of-the-Box BERTopic Integration**: Perform topic modeling
   seamlessly with the
   `BERTopic <https://maartengr.github.io/BERTopic/index.html>`__
   algorithm, leveraging embeddings from Large Language Models (LLMs).
-  **Language-Agnostic Capabilities**: Handle multiple languages, making
   it versatile for a wide range of linguistic contexts.
-  **Data Minimization and Process Optimization**: Limit the message
   features stored at the beginning of the analysis, supporting data
   minimization and reducing dataset size.

Installation
------------

Direct
~~~~~~

To install the required dependencies and run:

::

   pip install -r requirements.txt
   python main.py

Docker (recommended)
~~~~~~~~~~~~~~~~~~~~

Download the repository and then run:

::

   docker build -t pytopicgram .
   docker run -it pytopicgram

Usage
-----

You can run the entire pipeline through the ``main.py`` script. The
following command initiates the process, where messages in August 2024
from ``channels_sample.csv`` are downloaded, preprocessed, and analyzed
using topic modeling. An OpenAI key can be provided (at extra cost) to
generate topic descriptions in natural language.

Running the complete pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   cd pytopicgram
   python main.py \
       --api_id <TELEGRAM_API_ID> --api_hash <TELEGRAM_API_HASH> \
       --start_date 2024-08-01T00:00:00+00:00 \
       --end_date 2024-09-01T00:00:00+00:00 \
       --channels_file config/channels_sample.csv \
       --openai_key <OPENAI_KEY> \
       --description "Sample running, Aug 2024, using OpenAI API"

When running the ``main.py`` script, you can customize the behavior of
the pipeline using the following parameters:

+-------------------+--------------------------------------------------+
| Parameter Name    | Description                                      |
+===================+==================================================+
| ``--api_id``      | Your Telegram API ID, required to access the     |
|                   | Telegram API.                                    |
+-------------------+--------------------------------------------------+
| ``--api_hash``    | Your Telegram API Hash, required to access the   |
|                   | Telegram API.                                    |
+-------------------+--------------------------------------------------+
| ``--start_date``  | The start date for collecting messages, in ISO   |
|                   | 8601 format (e.g.,                               |
|                   | ``2024-08-01T00:00:00+00:00``).                  |
+-------------------+--------------------------------------------------+
| ``--end_date``    | The end date for collecting messages, in ISO     |
|                   | 8601 format (e.g.,                               |
|                   | ``2024-09-01T00:00:00+00:00``).                  |
+-------------------+--------------------------------------------------+
| ``                | Path to the CSV file containing the list of      |
| --channels_file`` | channels to be processed.                        |
+-------------------+--------------------------------------------------+
| ``--openai_key``  | Your OpenAI API key, used to generate topic      |
| (optional)        | descriptions in natural language.                |
+-------------------+--------------------------------------------------+
| ``--description`` | A description of the current run, useful for     |
| (optional)        | logging and tracking purposes.                   |
+-------------------+--------------------------------------------------+
| ``--craw          | Path to the output file where the crawled        |
| ler_output_file`` | messages will be saved.                          |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--preproce      | Path to the input file for the preprocessor.     |
| ssor_input_file`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--preproces     | Path to the output file where the preprocessed   |
| sor_output_file`` | messages will be saved.                          |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--preprocess    | List of features to be extracted during          |
| or_list_feature`` | preprocessing.                                   |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--preprocess    | List of channels to be included during           |
| or_list_channel`` | preprocessing.                                   |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--pre           | Limit on the number of messages to be            |
| processor_limit`` | preprocessed.                                    |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--pr            | Text to be used for preprocessing.               |
| eprocessor_text`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--remove_urls`` | Boolean flag to indicate whether URLs should be  |
| (optional)        | removed during preprocessing.                    |
+-------------------+--------------------------------------------------+
| ``                | Boolean flag to indicate whether emojis should   |
| --remove_emojis`` | be removed during preprocessing.                 |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--a             | Boolean flag to indicate whether descriptions    |
| dd_descriptions`` | should be added during preprocessing.            |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--c             | Boolean flag to indicate whether mentions should |
| apture_mentions`` | be captured during preprocessing.                |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``                | Path to the input file for metrics calculation.  |
| --metrics_input`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``-               | Path to the output file where the calculated     |
| -metrics_output`` | metrics will be saved.                           |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``                | Field to be used for metrics calculation.        |
| --metrics_field`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--me            | Number of neighbors to be considered for metrics |
| trics_neighbors`` | calculation.                                     |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``                | Alpha value to be used for metrics calculation.  |
| --metrics_alpha`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--metric        | Minimum threshold to be used for metrics         |
| s_min_threshold`` | calculation.                                     |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--nlp_input``   | Path to the input file for NLP processing.       |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--nlp_output``  | Path to the output file where the NLP results    |
| (optional)        | will be saved.                                   |
+-------------------+--------------------------------------------------+
| ``--nlp_text``    | Text to be used for NLP processing.              |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--nlp_split``   | Boolean flag to indicate whether the text should |
| (optional)        | be split during NLP processing.                  |
+-------------------+--------------------------------------------------+
| ``--extra         | Path to the input file for the topic extractor.  |
| ctor_input_file`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--extrac        | Path to the output file where the extracted      |
| tor_output_file`` | topics will be saved.                            |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--e             | Column to be used for topic extraction.          |
| xtractor_column`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--extra         | Number of topics to be extracted.                |
| ctor_num_topics`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--e             | Boolean flag to indicate whether OpenAI should   |
| xtractor_openai`` | be used for topic extraction.                    |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--extracto      | Number of documents to be used for OpenAI topic  |
| r_n_docs_openai`` | extraction.                                      |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--extract       | Ratio of samples to be used for topic            |
| or_sample_ratio`` | extraction.                                      |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--extracto      | Number of messages to be used for topic          |
| r_messages_used`` | extraction.                                      |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--vi            | Path to the model file for the topic viewer.     |
| ewer_model_file`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--vi            | Path to the input file for the topic viewer.     |
| ewer_input_file`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--viewe         | Path to the training file for the topic viewer.  |
| r_training_file`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``                | Column to be used for the topic viewer.          |
| --viewer_column`` |                                                  |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``                | Path to the output file where the topic viewer   |
| --viewer_output`` | results will be saved.                           |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--vi            | Number of topics to be displayed in the topic    |
| ewer_num_topics`` | viewer.                                          |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+
| ``--view          | Boolean flag to indicate whether visualizations  |
| er_generate_viz`` | should be generated in the topic viewer.         |
| (optional)        |                                                  |
+-------------------+--------------------------------------------------+

Using modules
~~~~~~~~~~~~~

In the ``examples`` folder, you can find examples of running individual
components of the pipeline. For instance, ``snowball.py`` demonstrates
how to use the snowball technique to gather messages from related
channels.

To run the snowball example, use the following command:

::

   cd pytopicgram
   python -m examples.snowball \
      --api_id <TELEGRAM_API_ID> --api_hash <TELEGRAM_API_HASH> \
      --start_date 2024-08-30T00:00:00+00:00 --end_date 2024-08-31T23:59:59+00:00 \
      --channels_file ./examples/snowball_channels_sample.csv \
      --output_channels_file ./examples/results/snowball_channels.csv \
      --output_messages_file ./examples/results/snowball_messages.json \
      --max_rounds 3

-  ``--api_id``: Your Telegram API ID, required to access the Telegram
   API.
-  ``--api_hash``: Your Telegram API Hash, required to access the
   Telegram API.
-  ``--start_date``: The start date for collecting messages, in ISO 8601
   format (e.g., ``2024-08-01T00:00:00+00:00``).
-  ``--end_date``: The end date for collecting messages, in ISO 8601
   format (e.g., ``2024-09-01T00:00:00+00:00``).
-  ``--channels_file``: Path to the CSV file containing the list of
   channels to be processed.
-  ``--output_channels_file``: Path to the output CSV file where the
   snowball channels will be saved.
-  ``--output_messages_file``: Path to the output JSON file where the
   collected messages will be saved.
-  ``--openai_key``: Your OpenAI API key, used to generate topic
   descriptions in natural language (optional).
-  ``--max_rounds``: The maximum number of rounds for the snowball
   process, determining how many iterations of related channel gathering
   will be performed.

More information
----------------

A short video on the use of ``pytopicgram``: `Watch the video on Google
Drive <https://drive.google.com/file/d/1jk_b95r5dGzeNiXisPdQr4zwT1tudUIe/view?usp=share_link>`__

Slides: `Check the
slides <https://drive.google.com/file/d/1jlfTSWeXoWEuSKQnzxAL6VZ_aNfQB-Rw/view?usp=sharing>`__

License
-------

This project is licensed under the Apache License 2.0 - see the
`LICENSE <LICENSE>`__ file for details.

Acknowledgments
---------------

This work was supported by the UDDOT project funded by the European
Media Information Fund (`EMIF <https://gulbenkian.pt/emifund/>`__)
managed by the Calouste Gulbenkian Foundation, the XAI-DISINFODEMICS
project (PLEC2021-007681) funded by MICIU/AEI/10.13039/501100011033 and
by European Union NextGenerationEU/PRTR, and The Social Observatory of
“la Caixa” Foundation.

.. |DOI:10.5281/zenodo.13863008| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13863008.svg
   :target: https://doi.org/10.5281/zenodo.13863008
