from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import wandb
# The token and secret are stored as environment variables
with open("secrets.json") as f:
    secrets = json.load(f)

EXPECTED_SECRET = secrets["expected_secret"]
EXPECTED_TOKEN = 'token'
 # example secret
api = wandb.Api()
class WebhookHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(self.headers)
        print(post_data)
        
        # Extracting the token, remove 'Bearer ' prefix
        authorization_header = self.headers.get('Authorization')
        token = authorization_header.split(' ')[-1] if authorization_header else None
        
        # Extracting the secret from the correct header
        secret = self.headers.get('X-Wandb-Signature')
        
        print("token = "+str(token)+" secret "+str(secret))
        
        if token == EXPECTED_TOKEN and secret == EXPECTED_SECRET:
            print("Received webhook payload:")
            print(post_data.decode())
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Webhook payload received')
            payload = json.loads(post_data)
            payload = payload["client_payload"]
            artifact = api.artifact(payload["artifact_version_string"], type='CIFAR_DATA')
            # You can add more processing logic here
            artifact.download()
        else:
            self.send_response(401)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Unauthorized')

if __name__ == '__main__':
    server_address = ('', 8090)
    httpd = HTTPServer(server_address, WebhookHandler)
    print('Webhook Receiver running on port 8090...')  # corrected port number in the print statement
    httpd.serve_forever()
