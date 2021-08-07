"""
In this funtion I run all of the tests needed to be run
"""
import time
from imports import *
from train import *
from os import environ
from torchvision.models import *


environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
hp = Help_Funcs()
# data, labels = hp.load_data(
#     IMG_SIZE=84,
#     directory="/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/data/raw/",
# )
# X_train, X_test, y_test, y_train = hp.split_data(labels, data)
# torch.save(X_train, "X_train.pt")
# torch.save(X_test, "X_test.pt")
# torch.save(y_test, "y_test.pt")
# torch.save(y_train, "y_train.pt")
labels = {0: "Not-Intrested", 1: "Intrested"}
labels = {"Not-Intrested": 0, "Intrested": 1}
X_train = torch.load("X_train.pt")
X_test = torch.load("X_test.pt")
y_test = torch.load("y_test.pt")
y_train = torch.load("y_train.pt")


class TL_Model(Module):
    def __init__(self, model, num_of_classes=1):
        super().__init__()
        self.model = model
        self.output = Linear(1000, num_of_classes)

    def forward(self, X):
        preds = self.model(X)
        preds = self.output(preds)
        preds = Sigmoid()(preds)
        return preds


epochs = 25
config["epochs"] = 25


# criterions
# optimizers
# lrs

config["batch_size"] = 64
config["criterion"] = HingeEmbeddingLoss
config["optimizer"] = AdamW
config["lr"] = 1e-05
epochs = 25
config["epoch"] = 25
config["epochs"] = 25
torch.cuda.empty_cache()
model = TL_Model(shufflenet_v2_x1_0(pretrained=True))
torch.cuda.empty_cache()
model = train_testing(
    X_train, y_train, X_test, y_test, model, f"Final-0", config=config,
)
# model = torch.load(f"./trained_models/model-ion.pt")
# paths = os.listdir("./data/test_data/")
# new_paths = []
# for path in paths:
#     new_paths.append(f"./data/test_data/{path}")
# hp.get_multiple_preds(paths=new_paths, model=model, IMG_SIZE=84)
