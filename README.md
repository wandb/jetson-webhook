# emea-sesame


This repo demonstrates using webhooks to train, re train and dowload wandb artifacts to and edge device (nvidia jetson) using wandb webhooks

![PXL_20230920_075839451.jpg](PXL_20230920_075839451.jpg)

A notebook providing high level 101 into what a webhook is doing  [here](https://colab.research.google.com/drive/1PdR7tzXbBp7HupE3pq633cHR7Qs-ub8x?usp=sharing)

Project demoing using wandb webhooks [here](https://wandb.ai/tiny-ml/quantized%20edge%20training)

appying the alias `retrain` on either data artifact or model artifact will dowload data to Nvidia Jetson and initialize re training.

webhook.py runs a super simple web server in python which listens for incoming signal -- in this case it has a real IP on a real device which serving from an Nvidia Jetson exposed to the public internet.

This can simply be run with `python webhook.py` from an Nvidia Jetson (or any other edge device configured in the same way)