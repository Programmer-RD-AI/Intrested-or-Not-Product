import random
import gc
from torchvision import transforms
from imports import *
from models.imports import *

config = {
    "IMG_SIZE": 84,
    "DATA_DIR": "/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/raw/",
    "PROJECT_NAME": "Intrested-or-Not-Intrested-In-A-Product-V2",
    "device": torch.device("cuda"),
    "lr": 0.001,
    "criterion": BCELoss(),
    "batch_size": 32,
    "epochs": 12,
    "optimizer": Adam,
    "input_size": (3, 84, 84),
    "output": 1,
    "output_ac": Sigmoid(),
}


"""
This 'helper_funtions.py' is going to be used to add small funtions \n
 like getting accuracy which I will need use almost all the time when I am training a model. so this a \n
 easy way for me to get all of the imports funtions in one place.
"""


class Help_Funcs(object):
    def transform_data(self, X, transformations):
        new_X = []
        for x in X:
            new_X.append(np.array(transformations(np.array(x))))
        X = new_X.copy()
        return torch.tensor(np.array(X))

    def get_faces(self, paths) -> dict or bool:
        """
        Gets In the paths of the images thatg need the faces croped into. (In)
        Then this Funtion uses *face_detection* lib to find the faces and save it. (Process)
        It Saves the Imgs in a order of indexs and Adds it to a dict. (Process)
        The return will usually look something like,{'img_path(the in)':[[1,..],['./out/1.png',..]]}. (Out)
        The return will be False if there is no Faces to be found. (Out)
        """
        idx = -1
        imgs_dict = {}
        for path in paths:
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                for face_location in tqdm(face_locations):
                    idx += 1
                    im = Image.open(fr"{path}")
                    left = face_location[3]
                    top = face_location[0]
                    right = face_location[1]
                    bottom = face_location[2]
                    im1 = im.crop((left, top, right, bottom))  # Croping into the Image
                    im1.save(
                        f"/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/{idx}.png"
                    )
                    # The below is the proccess of adding the idx and idx img to the dir and adding it to 'imgs_dict'
                    if path in list(imgs_dict.keys()):
                        imgs_dict[path][0].append(idx)
                        imgs_dict[path][1].append(
                            f"/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/{idx}.png"
                        )
                    else:
                        imgs_dict[path] = [
                            [idx],
                            [
                                f"/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/{idx}.png"
                            ],
                        ]
        return imgs_dict

    def __get_loss(self, model, X, y, criterion) -> float or int:
        """
        This is the main funtion but I am using this behind the get_loss funtion
        Takes in model,X,y,criterion,transformations. (In)
        I am using transformations also just for kind of checking just. (Comment)
        First this Funtion preds the X and gets the result and finds the loss using the criterion. (Process)
        The same thing happends with using the transformations. (Process)
        The output is a int of the mean (avg) of the 2 preds. (Out)
        """
        losses = []
        model.to(config["device"])
        preds = model(X.float().to(config["device"]))
        loss = criterion(
            preds.view(-1, 1).float().to(config["device"]),
            y.view(-1, 1).float().to(config["device"]),
        )
        losses.append(loss.item())
        loss = np.mean(np.array(losses))
        return loss

    def get_loss(self, model, X, y, criterion) -> float or int:
        """
        This is the main funtion this is using __get_loss()
        """
        losses = []
        loss = self.__get_loss(model, X, y, criterion)
        losses.append(loss)
        loss = np.mean(losses)
        return loss

    def __get_accuracy(self, model, X, y) -> float or int:
        """

        This is the Funtion which works behind the secens of get_accuracy(). (Comment)
        """
        accs = []
        with torch.no_grad():
            correct = -1
            total = -1
            model.to(config["device"])
            X = X.to(config["device"])
            preds = model(X.float())
            for idx in range(len(y)):
                if (
                    torch.round(torch.tensor([preds[idx]]).float()).float()
                    == torch.round(torch.tensor([int(y[idx])]).float()).float()
                ):
                    correct += 1
                total += 1
            acc = round(correct / total, 3)
        accs.append(acc)
        correct = -1
        total = -1
        preds = model(X.float())
        for idx in range(len(y)):
            if (
                torch.round(torch.tensor([preds[idx]]).float()).float()
                == torch.round(torch.tensor([int(y[idx])]).float()).float()
            ):
                correct += 1
            total += 1
        acc = round(correct / total, 3)
        accs.append(acc)
        return np.mean(acc)

    def get_accuracy(self, model, X, y) -> float or int:
        """
        This Funtion takes in model,X,y, and transformations. (In)
        This funtion call __get_accuracy() so this way of doing will save alot of time and lines of code. (Process)
        This Funtion outputs the accuracy of the model. (Out)
        """
        accs = []
        model.eval()
        accs.append(self.__get_accuracy(model, X, y))
        accs = np.array(accs)
        acc = np.mean(accs)
        return acc

    def get_accuracy_preds(self, preds, y) -> float or int:
        """
        This Funtion takes in preds and true labels(y). (In)
        I just iter over the preds and y and then see what is correct and then add 1 to correct and round(correct/total,3). (Process)
        This Funtion output the accuracy of the model. (Out)
        """
        with torch.no_grad():
            correct = -1
            total = -1
            for idx in range(len(y)):
                pred = torch.round(torch.tensor([preds[idx]])).float()
                y_batch = [int(y[idx])]
                y_batch = np.array(y_batch)
                y_batch = torch.from_numpy(y_batch)
                y_batch = torch.round(y_batch.float()).float()
                if pred == y_batch:
                    correct += 1
                total += 1
            acc = round(correct / total, 3)
        return acc

    def run_multiple_times(self, num_of_times, func, *args) -> list or int or float:
        """
        This Funtion takes in num_of_times,func,*args I take the *args to pass it to the func. (In)
        This Funtion does a for loop for how many times needed. and tries to mean if it can then it will return it if it cant it will just return the results. (Process)
        This Funtion Outputs a list or int or float. (Out)
        """
        results = []
        for _ in range(num_of_times):
            results.append(func(*args))
        try:
            results_mean = np.mean(np.array(results))
            return results_mean
        except:
            return results

    def __get_multiple_preds(self, paths, model, num_of_times=1, IMG_SIZE=84,) -> dict:
        """
        This funtion is kind the back of get_multiple_preds t
        """
        with torch.no_grad():
            preds = {}
            hp = Help_Funcs()
            faces_results = hp.get_faces(paths)
            for _ in range(num_of_times):
                imgs = []
                for key, val in zip(faces_results.keys(), faces_results.values()):
                    img = cv2.imread(val[1][0])
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    imgs.append(img)
                if imgs == []:
                    break
                preds_model = model(
                    torch.tensor(np.array(imgs))
                    .view(-1, 3, 84, 84)
                    .to(config["device"])
                    .float()
                )
                for key, val, pred in zip(
                    faces_results.keys(), faces_results.values(), preds_model
                ):
                    pred = int(round(float(pred)))
                    try:
                        preds[val[0][0]][0][int(pred)] += 1
                    except Exception as e:
                        preds[val[0][0]] = [{0: 0, 1: 0}, [key, val[1][0]]]
                        preds[val[0][0]][0][pred] += 1
            results = {}
            for idx, log in zip(preds.keys(), preds.values()):
                files = log[1]
                log = log[0]
                best_class = -1
                if log[0] < log[1]:
                    best_class = 1
                elif log[0] > log[1]:
                    best_class = 0
                img = cv2.imread(files[1])
                results[idx] = [
                    [best_class, files[0], files[1], img.tolist()],
                    files[0],
                    files[1],
                ]
            return results

    def get_multiple_preds(
        self,
        paths,
        model,
        labels_reverse={0: "no", 1: "yes"},
        num_of_times=100,
        IMG_SIZE=84,
    ) -> dict:
        """
        This funtion takes in a paths of the imgs that needs to be predicted multiple times.
        """
        model.eval()
        results = self.__get_multiple_preds(paths, model, num_of_times, IMG_SIZE)
        return results

    def load_data(
        self,
        directory="/home/indika/Programming/Projects/Python/Artifical-Intelligence/PyTorch/Competions/Emo-Pro/other/Intrested-or-Not-Product-V2/raw/",
        IMG_SIZE=84,
        transformations=transforms.Compose([transforms.ToTensor()]),
    ) -> list and dict:
        """
        loads data using cv2
        """
        main_dir = directory
        data = []
        labels = {}
        labels_idx = -1
        for directory in os.listdir(directory):
            labels_idx += 1
            labels[main_dir + directory + "/"] = [labels_idx, -1]
        for label in tqdm(labels.keys()):
            for file in os.listdir(label):
                try:
                    labels[label][1] += 1
                    img = cv2.imread(label + file)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(
                        [
                            np.array(transformations(np.array(img / 255.0))),
                            labels[label][0],
                        ]
                    )
                except:
                    print(file)
        return data, labels

    def split_data(self, labels, data):
        """
        the funtion splits the data to train and test splits
        """
        for _ in range(100):
            np.random.shuffle(data)
        X = []
        y = []
        for d in data:
            X.append(d[0])
            y.append(d[1])
        VAL_SPLIT = 2500
        X_train = np.array(X[:-VAL_SPLIT])
        y_train = np.array(y[:-VAL_SPLIT])
        X_test = np.array(X[-VAL_SPLIT:])
        y_test = np.array(y[-VAL_SPLIT:])
        for key, val in zip(labels.keys(), labels.values()):
            print("*" * 50)
            print(key)
            print(val)
            print("*" * 50)
        return (
            torch.tensor(X_train),
            torch.tensor(X_test),
            torch.tensor(y_test),
            torch.tensor(y_train),
        )
