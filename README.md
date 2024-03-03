# Financial Text Sentiment Analysis App

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![terminal](https://img.shields.io/badge/windows%20terminal-4D4D4D?style=for-the-badge&logo=windows%20terminal&logoColor=white)
![git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)
![markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)
![Made with Docker](https://img.shields.io/badge/Made%20with-Docker-blue?style=for-the-badge&logo=Docker)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![CSS](https://img.shields.io/badge/CSS-1572B6?style=for-the-badge&logo=css3&logoColor=white)

## Overview
This application developed with FastAPI, JavaScript and CSS specializes in categorizing financial texts into positive, negative, or neutral sentiments. It leverages data extracted from the [Bitcoin channel on Reddit's API](https://www.reddit.com/r/Bitcoin/) to analyze and visualize sentiment progression alongside actual cryptocurrency price trends. Additionally, the application offers a feature to classify user-input text into the aforementioned sentiment categories.

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

To get a local copy of this project up and running on your computer, follow these simple steps.

### Prerequisites

- Ensure you have Git installed on your machine. If not, follow the [Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your specific operating system.

### Cloning the Repository

1. **Open your terminal or command prompt**. On Windows, you might use Command Prompt or PowerShell, and on macOS or Linux, you'll use the Terminal.

2. **Navigate to the directory** where you want to clone the repository. Use the `cd` command to change directories. For example, to change to the Documents directory, you would type:
   ```bash
   cd Documents
   ```

3. **Clone the repository** by running the following command in your terminal. Replace `YourUsername` with your actual GitHub username if the repository is private and requires your GitHub credentials for cloning.
   ```bash
   git clone https://github.com/YourUsername/project-repository-name](https://github.com/pablo-git8/FinSentNewsNLP).git
   ```

4. **Navigate to the project directory** by running:
   ```bash
   cd FinSentNewsNLP
   ```
Now you have a local copy of the project, and you can start exploring the code and contributing to the project.


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
