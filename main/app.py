from fastapi import FastAPI
import praw
import logging
import sys
import os
import json
from get_data import data

app = FastAPI()
app.include_router(data)
logging_dir = "/logs"
logging_file = os.path.join(logging_dir, "errors.log")
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
# TODO change the level to INFO once we are done with dev
logging.basicConfig(filename=logging_file, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
