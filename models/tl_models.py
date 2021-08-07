from imports import *

"""
This TL_Model takes a pretrained TL Model TL = Transfer Learning.
And it add a linear layer to the end.
The TL_Model_2 is kind of a testing model I am using the pretrained model and then predicting and then converting the preds to the a flatten image and then passing the image again to the model.
"""


class TL_Model(Module):
    def __init__(self, model):
        super().__init__()
        output = config["output"]
        output_ac = config["output_ac"]
        self.model = model(pretrained=True)
        self.output = Linear(1000, output)
        self.output_ac = output_ac

    def forward(self, X):
        X = X.view(-1, 3, 84, 84)
        preds = self.model(X)
        preds = self.output(preds)
        preds = self.output_ac(preds)
        return preds


class TL_Model_2(Module):
    def __init__(self, model):
        super().__init__()
        output = config["output"]
        output_ac = config["output_ac"]
        self.model_1 = model(pretrained=True)# resnet18
        self.output_1 = Linear(1000, 3 * 84 * 84)
        self.model_2 = model(pretrained=True)
        self.output_2 = Linear(1000, output)
        self.output_ac = output_ac

    def forward(self, X):
        X = X.view(-1, 3, 84, 84)
        preds = self.model_1(X)
        preds = self.output_1(preds)
        preds = preds.view(-1, 3, 84, 84)
        preds = self.model_2(preds)
        preds = self.output_2(preds)
        preds = self.output_ac(preds)
        return preds


