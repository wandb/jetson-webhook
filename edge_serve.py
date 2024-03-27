from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import wandb
import os
import subprocess
from training.qunantized_train import TrainModel, SmallQuantizedCNN
# The token and secret are stored as environment variables
with open("secrets.json") as f:
    secrets = json.load(f)

EXPECTED_SECRET = secrets["wandb-secret"]
EXPECTED_TOKEN = 'X-WANDB-Signature'
os.environ['WANDB_API_KEY'] = secrets["wandb-api-key"]
 # example secret
api = wandb.Api()
class WebhookHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        print(f'wandb webhook headers {self.headers}')
        print(f'post data {post_data}')
        
        # Extracting the token, remove 'Bearer ' prefix
        secret = self.headers.get('Authorization')
        print(f"authorization header = {secret}")
        secret = secret.split(' ')[-1] if secret else None
        
        # Extracting the secret from the correct header
        token = self.headers.get('X-Wandb-Signature')
        
        print("token = "+ EXPECTED_TOKEN +" secret " +str(secret))
        
        if secret == EXPECTED_SECRET:
            print("Received webhook payload:")
            print(post_data.decode())
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Webhook payload received')
            payload = json.loads(post_data)
            payload = payload["client_payload"]
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
            self.send_response(401)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Unauthorized')

if __name__ == '__main__':
    server_address = ('', 8090)
    httpd = HTTPServer(server_address, WebhookHandler)
    print('Webhook Receiver running on port 8090')  # corrected port number in the print statement
    httpd.serve_forever()
