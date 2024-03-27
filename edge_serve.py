from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import wandb
import os
import subprocess
import hashlib
import hmac
import base64
from training.qunantized_train import TrainModel, SmallQuantizedCNN

# The token and secret are stored as environment variables
with open("secrets.json") as f:
    secrets = json.load(f)

EXPECTED_SECRET = secrets["wandb-secret"]
os.environ['WANDB_API_KEY'] = secrets["wandb-api-key"]
# example secret
api = wandb.Api()

def verify_signature(secret_key, hex_received_signature):
    payload = '{}'  # The actual payload that was signed, ensure it matches exactly
    hmac_signature = hmac.new(secret_key.encode(), payload.encode(), hashlib.sha256).digest()

    # Convert the received hex signature to binary
    binary_received_signature = bytes.fromhex(hex_received_signature)

    # Direct comparison of the binary formats
    is_valid = hmac_signature == binary_received_signature
    return is_valid


class WebhookHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        print(f'wandb webhook headers {self.headers}')
        print(f'post data {post_data}')

        # Extracting the token, remove 'Bearer ' prefix
        secret = self.headers.get('Authorization')
        secret = secret.split(' ')[-1] if secret else None
        print(f"authorization header = {secret}")

        # Extracting the X-Wandb-Signature from the header
        signature = self.headers.get('X-Wandb-Signature')

        # Verify the HMAC signature
        if verify_signature(EXPECTED_SECRET,signature):
            print("Signature verified.")
            print("Received webhook payload:")
            print(post_data.decode())
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Webhook payload received')
            payload = json.loads(post_data)
            payload = payload.get("client_payload", {})
            if 'artifact_collection_name' in payload:
                if payload['artifact_collection_name'] == 'CIFAR_10':
                    artifact = api.artifact(payload["artifact_version_string"], type='CIFAR_DATA')
                    artifact.download()
                elif payload['artifact_collection_name'] == "quantized_model":
                    artifact = api.artifact(payload["artifact_version_string"], type='model')
                    artifact.download()
                    # retrain model as a subprocess with env vars
                    model = SmallQuantizedCNN()
                    train = TrainModel(model)
                    train.wandb_login()
                    train.wandb_init()
                    train.create_wandb_artifact()
                    train.train()

                # You can add more processing logic here
        else:
            print("Signature verification failed.")
            self.send_response(401)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Unauthorized')

if __name__ == '__main__':
    server_address = ('', 8090)
    httpd = HTTPServer(server_address, WebhookHandler)
    print('Webhook Receiver running on port 8090')
    httpd.serve_forever()

