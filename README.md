# Financial Text Sentiment Analysis App

## Overview
This application specializes in categorizing financial texts into positive, negative, or neutral sentiments. It leverages data extracted from the [Bitcoin channel on Reddit's API](https://www.reddit.com/r/Bitcoin/) to analyze and visualize sentiment progression alongside actual cryptocurrency price trends. Additionally, the application offers a feature to classify user-input text into the aforementioned sentiment categories.

## Data and Models
The sentiment analysis is powered by baseline logistic regression models and the more sophisticated FinBERT models, which are designed for financial text. The initial models were trained using the [Financial Phrasebank dataset](https://huggingface.co/datasets/financial_phrasebank), known for its human-annotated sentiment labels.

For comprehensive analysis, both small and large versions of FinBERT were employed, accommodating texts of varying lengths. Training datasets include sentiment-labeled sentences, specifically:
- Sentences_50Agree.txt
- Sentences_66Agree.txt
- Sentences_75Agree.txt
- Sentences_AllAgree.txt

These are stored within the `data/raw` directory. The trained models and associated vectorizers and tokenizers are organized as follows:

- Models: `models/`
  - Large and small FinBERT models
  - Logistic regression models for different agreement levels
- Tokenizers: `tokenizers/`
  - Large FinBERT tokenizer
- Vectorizers: `vectorizers/`
  - Small FinBERT vectorizer
  - Various agreement level vectorizers

The Large FinBERT model was fine-tuned using a specialized dataset containing Bitcoin and cryptocurrency-related sentences, enhancing its performance for this domain.

## Development and Validation
The development journey, including LLM and NLP training and validation processes, is thoroughly documented in Jupyter notebooks located in the `notebooks/` directory. Specifically, the `Assignment-04-Financial news analysis with NLP.ipynb` notebook provides insights into the initial phases of model development and training.

The fine-tuning process, utilizing Google Colab's GPU capabilities, is detailed in `Assignment-04-FineTuning.ipynb`. In this notebook, we fine-tuned the model to enhance its performance on cryptocurrency conversations. To address the challenge of class imbalance and to better adapt the model to the nuances of cryptocurrency discourse, we incorporated additional data from two specific datasets:

- We enriched our training data with 100 English rows labeled as negative from the [Bitcoin Tweets Sentiment Kaggle dataset](https://huggingface.co/datasets/ckandemir/bitcoin_tweets_sentiment_kaggle/viewer/default/train?p=1). This was crucial for dealing with the class imbalance issue.
- Additionally, we utilized the [Alpaca Bitcoin Sentiment Dataset](https://huggingface.co/datasets/Andyrasika/alpaca-bitcoin-sentiment-dataset/viewer/default/train?p=1) to obtain bitcoin-related financial texts, further tailoring our model to the specific language and sentiment found in cryptocurrency discussions.

These targeted datasets allowed us to refine the model's understanding of cryptocurrency-related sentiments, significantly improving its predictive accuracy and relevance in financial text analysis within the crypto domain.

## Getting Started

### Using Poetry
The project dependencies are managed using Poetry. To install the dependencies and run the application, use the following commands:

```bash
poetry install
cd /main
poetry run uvicorn app:app --reload
```

### Using Docker
For convenience, a Dockerfile is provided within the `docker/Docker` directory. To build and run the application using Docker, navigate to the root directory of the project and execute the following commands:

```bash
docker build -t sentiment-analysis-app ./docker/Docker
docker run -p 8000:8000 sentiment-analysis-app
```

This will build the Docker image with the tag `sentiment-analysis-app`, using the Dockerfile located in `docker/Docker`. After building the image, the `docker run` command starts a container from this image, mapping port 8000 of the container to port 8000 on your host, allowing you to access the application as needed.


## Contributions
We encourage contributions and feedback on our project. Feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
