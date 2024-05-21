# Use an official Python runtime as a parent image
FROM python:3.11.9-slim-bookworm

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --use-deprecated=legacy-resolver -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# # Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "api_call:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
