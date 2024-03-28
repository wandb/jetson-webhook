from training.qunantized_train import TrainModel, SmallQuantizedCNN
import json
import os

with open("secrets.json") as f:
    secrets = json.load(f)

EXPECTED_SECRET = secrets["wandb-secret"]
os.environ['WANDB_API_KEY'] = secrets["wandb-api-key"]
model = SmallQuantizedCNN()
train = TrainModel(model)
train.wandb_login()
train.wandb_init()
train.create_wandb_artifact()
train.train()
