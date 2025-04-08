import requests

url = "http://127.0.0.1:8000/api/check_plagiarism/"
headers = {"Content-Type": "application/json"}
data = {
    "input_text": "This is a test.",
    "reference_text": "This is another test."
}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # Should return the API response
