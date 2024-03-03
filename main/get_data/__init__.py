from datetime import datetime
import praw
import sys
import os
import json
import logging
from fastapi import APIRouter, status
import sqlite3 as sql
from enum import Enum
import time
from transformers import \
    AutoModelForSequenceClassification, AutoTokenizer, \
    BertForSequenceClassification, BertTokenizer, BertForSequenceClassification, \
    AdamW, BertConfig, get_linear_schedule_with_warmup

import torch

data = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/get_data",
    tags=["data"],
)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the root folder by going up two levels from the current directory
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Path to the config folder relative to the root folder
config_dir = os.path.join(root_dir, "config")

# Path to the prepared folder relative to the root folder
prepared_dir = os.path.join(root_dir, "data", "prepared")

try:
    with open(os.path.join(config_dir, "key.json"), "r") as file:
        client_data = json.load(file)
except Exception as e:
    logging.error(
        f"Could not load json data from config folder! "
        f"Are you sure you have created the json file containing your API information? Error: {e}")
    sys.exit(1)
try:
    reddit = praw.Reddit(
        client_id=client_data["client_id"],
        client_secret=client_data["client_secret"],
        user_agent=client_data["user_agent"]
    )
except Exception as e:
    logging.error(f"Could not connect to the Reddit API! Error: {e}")
    sys.exit(1)

# Check if the prepared folder exists and create it if not
if not os.path.exists(prepared_dir):
    try:
        os.makedirs(prepared_dir)
    except Exception as e:
        logging.error(f"Couldn't create the 'prepared' directory: {e}")
        sys.exit(1)


class Type(Enum):
    post = 0
    comment = 1


# <editor-fold desc="My functions">
def get_conn() -> sql.Connection:
    """
    Initializes the connection to the SQLite database and creates the necessary tables if they don't exist
    :return: Connection object
    """
    try:
        con = sql.connect(os.path.join(prepared_dir, "fin_nlp.db"))
        cursor = con.cursor()
        sql_create_table = ("CREATE TABLE IF NOT EXISTS fin_nlp_data_bitcoin"
                            "(id INT PRIMARY KEY AUTOINCREMENT , main_text TEXT, "
                            "date_creation DATETIME, upvotes INT, downvotes INT, tokenized_text TEXT, "
                            "sentiment VARCHAR(255))")
        cursor.execute(sql_create_table)
        con.commit()
    except sql.DatabaseError as error:
        logging.error(f"Could not connect to the database: {error}")
        sys.exit(1)
    return con


def get_latest_date(conn: sql.Connection) -> str | None:
    """
    Gets the latest date in the database
    :param conn: Connecion to the SQLite database
    :return: The latest date of creation or None if empty table
    """
    cursor = conn.cursor()
    sql_query = "SELECT date_creation FROM fin_nlp_data_bitcoin WHERE type = 0 ORDER BY date_creation DESC LIMIT 1"
    cursor.execute(sql_query)
    latest_date = cursor.fetchone()
    return latest_date[0] if latest_date else None


def insert_row(post_reddit, connection: sql.Connection, date_creation: str, type_row=Type.post) -> None:
    # TODO To optimize the code, we should go for an insertion in batches instead of one insert per row
    try:
        cursor = connection.cursor()
        if type_row == Type.post:
            content = post_reddit.selftext
        elif type_row == Type.comment:
            content = post_reddit.body
        else:
            raise Exception(f"Unknown type passed as an argument.")
        sql_insert = f"INSERT INTO fin_nlp_data_bitcoin VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        data_sql = (post_reddit.id, content, date_creation,
                    post_reddit.score, post_reddit.downs, None, None, 0)
        cursor.execute(sql_insert, data_sql)
        connection.commit()
    # This exception keeps the code going even if there is an error inserting one row
    except Exception as exception:
        logging.error(f"Error inserting data in the SQL database: {exception}")


def get_input_ids_and_attention_mask_chunk(tokens):
    """
    Splits the input_ids and attention_mask tensors into chunks, ensuring that each chunk, except possibly the last,
    has a length of chunksize - 2 to accommodate for the addition of special tokens. Special tokens ([CLS] and [SEP])
    are prepended and appended to each chunk respectively. Chunks are padded with zeros to maintain uniform size.

    Returns:
        Tuple containing two lists of tensors, one for the chunked input_ids and the other for the chunked attention masks.
    """
    chunksize = 512
    # Split the tokens into chunks, leaving space for special tokens
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    attention_mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

    # Iterate through each chunk to add special tokens and padding
    for i in range(len(input_id_chunks)):
        # Prepend [CLS] and append [SEP] tokens (101 and 102 respectively)
        input_id_chunks[i] = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])
        # Prepend and append attention mask bits for special tokens
        attention_mask_chunks[i] = torch.cat([
            torch.tensor([1]), attention_mask_chunks[i], torch.tensor([1])
        ])

        # Calculate padding length
        pad_length = chunksize - input_id_chunks[i].shape[0]
        # Apply padding if necessary
        if pad_length > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_length)
            ])
            attention_mask_chunks[i] = torch.cat([
                attention_mask_chunks[i], torch.Tensor([0] * pad_length)
            ])

    return input_id_chunks, attention_mask_chunks


def get_sentiment(text: str) -> str:
    # Initialize the tokenizer and model with a pre-trained BERT model specifically fine-tuned for financial texts
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

    # Mapping for evaluation
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    # Tokenize the text, specifying not to add special tokens and to return PyTorch tensors
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

    # Split the tokenized input_ids and attention_mask into chunks of size 510
    input_id_chunks = tokens['input_ids'][0].split(510)
    attention_mask_chunks = tokens['attention_mask'][0].split(510)

    # Get the processed chunks for input_ids and attention_mask
    input_id_chunks, attention_mask_chunks = get_input_ids_and_attention_mask_chunk(tokens)

    # Stack the chunks to form tensors
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(attention_mask_chunks)

    # Prepare the input dictionary for the model
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    }

    # Pass the inputs through the model to get the outputs
    outputs = model(**input_dict)

    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)

    # Compute the mean probabilities across all chunks
    mean_probabilities = probabilities.mean(dim=0)

    # Map the index to its corresponding label
    predicted_class_index = torch.argmax(mean_probabilities).item()
    predicted_class_label = label_map[predicted_class_index]

    # Print the predicted class label
    print(f"Predicted class: {predicted_class_label}")
    return predicted_class_label


def insert_sentiment(con, sentiment, id_row):
    try:
        cursor = con.cursor()
        sql_query = "UPDATE fin_nlp_data_bitcoin SET sentiment = ? WHERE id = ?"
        params = (sentiment, id_row)
        cursor.execute(sql_query, params)
        con.commit()
    # This exception allows the insertions to continue if one fails
    except Exception as exception:
        logging.error(f"An unexpected error occured: {exception}")
# </editor-fold>


@data.get("/data")
def get_data() -> bool:
    """
    This function will gather data from the Reddit API and store it in a SQLlite database
    :return: True if the request was successful, False otherwise
    """
    con = get_conn()
    latest_date = get_latest_date(con)
    logging.debug(f"Latest date:{latest_date}")
    subreddit_name = "Bitcoin"
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.top(limit=10000, time_filter="year"):
        # If the text is empty, the post is probably an image or video we skip to the next post
        if post.selftext == "":
            continue
        date_creation = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        if date_creation == latest_date:
            logging.info(f"This post ({post.id}) already exists in the database, stopping insert process.")
            break
        insert_row(post, con, date_creation)
        logging.info(f"Inserting the post: {post.id}")
        time.sleep(1)
        # print("Text: ", post.selftext)
        # print("Upvotes: ", post.score)
        # print("Downvotes: ", post.downs)
        # print("Comments: ")
        # Get all the comments of the current post
        # post.comments.replace_more(limit=None)
        # for comment in post.comments.list():
        #     date_creation_comment = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        #     insert_row(comment, con, date_creation_comment, Type.comment)
        #     logging.info(f"Inserting the comment: {comment.id}")
        # print(" - ", comment.body)
        # print("    Upvotes: ", comment.score)
        # print("    Downvotes: ", comment.downs)
    return True


@data.post("/bitcoin_data")
def post_bitcoin_data() -> list | str:
    """
    This function is used to get the data from the Bitcoin Reddit database and return it as a dictionnary
    :return: A dictionary containing the data from the Bitcoin Reddit database or the error message
    """
    # Get the number of postive, negative and neutral posts towards investing in Bitcoin
    # grouped by financial quarter of each year
    sql_query = ("""
        SELECT COUNT(*), 
               (CAST(STRFTIME('%Y', date_creation) AS INTEGER) || '-' || 
                CAST(((CAST(STRFTIME('%m', date_creation) AS INTEGER) - 1) / 3 + 1) AS TEXT)) AS quarter,
               sentiment
        FROM fin_nlp_data_bitcoin
        GROUP BY quarter, sentiment
        ORDER BY quarter
    """)
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data_bitcoin = cursor.fetchall()
        return data_bitcoin
    except Exception as exception:
        logging.error(f"An error occured: {exception}")
        return str(exception)


@data.get("/preparation")
def preparation() -> bool:
    conn = get_conn()
    cursor = conn.cursor()
    sql_query = "SELECT id, main_text FROM fin_nlp_data_bitcoin WHERE sentiment IS NULL"
    cursor.execute(sql_query)
    data_to_prepare = cursor.fetchall()
    for row in data_to_prepare:
        sentiment = get_sentiment(row[1])
        insert_sentiment(conn, sentiment, row[0])
    return True
