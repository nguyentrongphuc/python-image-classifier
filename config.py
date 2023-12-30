import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = 'image-classifier-dataset.pth'

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]

SIZE = 256,256

BATCH_SIZE = 32
