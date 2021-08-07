from models.imports import *
from imports import *

"""
The Input is a flatten image so if the img shape is 3,112,112 the input for this model if 3*112*112
"""

class Clf(Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Intialization
        """
        output = config["output"]
        self.activation = LeakyReLU()
        self.dropout_linear = Dropout()
        self.linear1 = Linear(3 * 84 * 84, 512)
        self.linear1_batch_norm = BatchNorm1d(512)
        self.linear2 = Linear(512, 1024)
        self.linear2_batch_norm = BatchNorm1d(1024)
        self.linear3 = Linear(1024, 512)
        self.linear3_batch_norm = BatchNorm1d(512)
        self.linear4 = Linear(512, output)
        self.linear4_activation = config["output_ac"]

    def forward(self, X):
        """
        Going through the Layers and predicting
        """
        X = X.view(-1, 3 * 84 * 84)
        preds = self.activation(
            self.dropout_linear(self.linear1_batch_norm(self.linear1(X)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear2_batch_norm(self.linear2(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear3_batch_norm(self.linear3(preds)))
        )
        preds = self.linear4_activation(self.linear4(preds))
        return preds
