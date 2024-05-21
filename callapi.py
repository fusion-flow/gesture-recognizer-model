import requests

# Replace the URL with the actual URL where your FastAPI server is running
api_url = "http://127.0.0.1:8000/gesture-recognizer/process-image"

# for loop to send multiple images
for i in range(1, 9):
    # Replace the file_path with the path to the image you want to send
    file_path = "sample images/sample_" + str(i) + ".jpg"
    print(file_path)

    # Create a dictionary with the file to be uploaded
    files = {"file": open(file_path, "rb")}

    # Make the POST request
    response = requests.post(api_url, files=files)

    # Print the response
    print(response.json(), file_path)
