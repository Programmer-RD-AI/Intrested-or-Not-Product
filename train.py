"""
This file has the training proccess
"""
import gc
from imports import *


def train_testing(X_train, y_train, X_test, y_test, model, name, config=config):
    # try:
    hp = Help_Funcs()
    model = model
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
    criterion = BCELoss()
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    torch.cuda.empty_cache()
    wandb.init(project=config["PROJECT_NAME"], name=name, sync_tensorboard=True)
    torch.cuda.empty_cache()
    for _ in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        for idx in range(0, len(X_train), batch_size):
            torch.cuda.empty_cache()
            X_batch = X_train[idx : idx + batch_size].view(-1, 3, 84, 84).to(device)
            y_batch = y_train[idx : idx + batch_size].to(device)
            model.to(device)
            preds = model(X_batch.float())
            loss = criterion(
                preds.to(device).float().view(-1, 1),
                y_batch.view(-1, 1).float().to(device),
            )
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        wandb.log({"loss": loss.item()})
        wandb.log({"val_loss": hp.get_loss(model.to("cpu"), X_test, y_test, criterion)})
        wandb.log({"accuracy": hp.get_accuracy_preds(preds, y_batch) * 100})
        wandb.log({"val_accuracy": hp.get_accuracy(model, X_test, y_test) * 100})
        model.to(device)
    paths = os.listdir("./data/test_data/")
    new_paths = []
    for path in paths:
        new_paths.append(f"./data/test_data/{path}")
    hp.get_multiple_preds(paths=new_paths, model=model, IMG_SIZE=84)
    paths = os.listdir("./out/")
    for path in paths:
        wandb.log({f"img/{path}": wandb.Image(cv2.imread(f"./out/{path}"))})
    wandb.finish()
    torch.save(model, f"./trained_models/model-ion.pt")
    torch.save(model, f"./trained_models/model-ion.pth")
    torch.save(model.state_dict(), f"./trained_models/model-ion-sd.pt")
    torch.save(model.state_dict(), f"./trained_models/model-ion-sd.pth")
    return model


# except Exception as e:
#     print(e)
#     torch.cuda.empty_cache()

