import praw
import sys
import os
import json
import logging
from fastapi import APIRouter, status
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


@data.get("/data")
def get_data() -> bool:
    """
    This function will gather data from the Reddit API and store it in a SQLlite database
    :return: True if the request was successful, False otherwise
    """
    subreddit = reddit.subreddit('ChicagoRealEstate')
    # Get the 5 first posts while filer HOT is active
    for post in subreddit.hot(limit=5):
        print("Text: ", post.selftext)
        print("Upvotes: ", post.score)
        print("Downvotes: ", post.downs)
        print("Comments: ")
        # Get all the comments of the current post
        post.comments.replace_more(limit=None)
        for comment in post.comments.list():
            print(" - ", comment.body)
            print("    Upvotes: ", comment.score)
            print("    Downvotes: ", comment.downs)
    return True
