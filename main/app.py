from fastapi import FastAPI
import praw
import logging
import sys
import os
import json
from get_data import data

app = FastAPI()
app.include_router(data)
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the root folder by going up one level from the current directory
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
logging_dir = os.path.join(root_dir, "logs")
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
logging_file = os.path.join(logging_dir, "logs.log")
# TODO change the level to INFO once we are done with dev
logging.basicConfig(filename=logging_file, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application started.")
print("Application started.")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
