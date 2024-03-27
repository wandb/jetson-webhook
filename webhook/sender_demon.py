import json
import requests

# URL of the server to send the POST request to
url = "http://192.168.1.200:8090"

# Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer token',  # Replace 'token' with your actual token
    'wandb-secret': 'a-secret-key'  # Replace 'your_secret' with your actual secret
}

# JSON Payload
payload = {
    "client_payload": {
        "artifact_version_string": "your_artifact_version_string"  # Replace with your actual artifact version string
    }
}

# Convert the payload to a JSON-formatted string
payload_json = json.dumps(payload)

# Send POST request
response = requests.post(url, headers=headers, data=payload_json)

# Print the response
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
