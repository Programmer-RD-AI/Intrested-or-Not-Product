# this is just a kind of a backup file
from torchvision import models
from imports import *
from train import *
from torchvision.models import *

hp = Help_Funcs()
data, labels = hp.load_data()
import json

with open(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/labels.json",
    "w",
) as file:
    json.dump(labels, file)

with open(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/labels.json",
    "r",
) as file:
    labels = json.load(file)
X_train, X_test, y_test, y_train = hp.split_data(labels, data)
torch.save(
    X_train,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_train.pt",
)
torch.save(
    X_train,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_train.pth",
)
torch.save(
    y_train,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_train.pt",
)
torch.save(
    y_train,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_train.pth",
)
torch.save(
    X_test,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_test.pt",
)
torch.save(
    X_test,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_test.pth",
)
torch.save(
    y_test,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_test.pt",
)
torch.save(
    y_test,
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_test.pth",
)

X_train = torch.load(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_train.pt"
)
y_train = torch.load(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_train.pt"
)
X_test = torch.load(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/X_test.pt"
)
y_test = torch.load(
    "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/cleaned/y_test.pt"
)

# model = Clf()
# train_testing(X_train, y_train, X_test, y_test, model, "Clf-0", config=config)
# model = CNN()
# train_testing(X_train, y_train, X_test, y_test, model, "CNN-0", config=config)
# model = Clf_and_Conv1d()
# train_testing(
#     X_train, y_train, X_test, y_test, model, "Clf_and_Conv1d-0", config=config
# )
# model = TL_Model(alexnet)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-alexnet", config=config,
# )
# model = TL_Model(squeezenet1_0)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-squeezenet1_0", config=config,
# )
# model = TL_Model(squeezenet1_1)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-squeezenet1_1", config=config,
# )
# model = TL_Model(googlenet)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-googlenet", config=config,
# )
# model = TL_Model(resnext50_32x4d)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-resnext50_32x4d", config=config,
# )
# model = TL_Model(wide_resnet50_2)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-wide_resnet50_2", config=config,
# )
# model = TL_Model(mnasnet0_5)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-mnasnet0_5", config=config,
# )
# model = TL_Model(mobilenet_v2)
# train_testing(
#     X_train, y_train, X_test, y_test, model, f"TL_Model-mobilenet_v2", config=config,
# )
# model = TL_Model(shufflenet_v2_x0_5)
# train_testing(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     model,
#     f"TL_Model-shufflenet_v2_x0_5",
#     config=config,
# )

config["batch_size"] = 32
config["optimizer"] = AdamW
config["lr"] = 0.001
config["criterion"] = BCELoss()
config["epochs"] = 12
hp = Help_Funcs()
data, labels = hp.load_data(
    directory="/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Intrested-or-Not-Product-V2/data/raw/"
)
X_train, X_test, y_test, y_train = hp.split_data(labels, data)
model = TL_Model(models.shufflenet_v2_x0_5)
model = train_testing(
    X_train, y_train, X_test, y_test, model, f"TL_Model-{config['epochs']}", config=config,
)
paths = os.listdir("./data/test_data/")
new_paths = []
for path in paths:
    new_paths.append(f"./data/test_data/{path}")
hp.get_multiple_preds(paths=new_paths, model=model)

torch.cuda.empty_cache()
model = TL_Model_2(shufflenet_v2_x0_5)
torch.cuda.empty_cache()
train_testing(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    f"TL_Model_2-shufflenet_v2_x0_5",
    config=config,
)
torch.cuda.empty_cache()
model = TL_Model_2(mnasnet0_5)
torch.cuda.empty_cache()
train_testing(
    X_train, y_train, X_test, y_test, model, f"TL_Model_2-mnasnet0_5", config=config,
)
torch.cuda.empty_cache()
