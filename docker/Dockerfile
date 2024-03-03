# Use the official Python 3.11-slim image from Docker Hub
FROM python:3.11-slim

# Set environment variables
ENV APP_DIR /opt/app
ENV BDI_LOCAL_DIR /opt/data

# Set the working directory in the container
WORKDIR $APP_DIR

# Install Poetry
RUN pip install poetry

# Copy the contents of the current directory into the container at /opt/app
COPY . $APP_DIR

# Install dependencies using Poetry
RUN poetry install

# Change the working directory to /opt/app/finsent_api
WORKDIR $APP_DIR/finsent_api

# Expose the necessary port
EXPOSE 8000

# Specify the command to run your application
CMD ["poetry", "run", "uvicorn", "bdi_api.app:app", "--host", "0.0.0.0"]