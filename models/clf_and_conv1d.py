from imports import *
"""
In this File I am just kind of testing to see if a Clf_and_Conv1d is better than a CNN 
I am using Conv1d Layers Instend of Conv2d

"""

class Clf_and_Conv1d(Module):
    def __init__(self) -> None:
        """
        Intialization
        """
        super().__init__()
        output = config["output"]
        self.activation = LeakyReLU()
        self.max_pool2d = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_conv = Dropout()
        self.conv1 = Conv1d(3, 8, (5, 5), (1, 1),)
        self.conv1_batchnorm = BatchNorm2d(8)
        self.conv2 = Conv1d(8, 16, (5, 5), (1, 1))
        self.conv2_batchnorm = BatchNorm2d(16)
        self.dropout_linear = Dropout()
        self.linear1 = Linear(16 * 18 * 18, 128)
        self.linear1_batchnorm = BatchNorm1d(128)
        self.linear2 = Linear(128, 256)
        self.linear2_batchnorm = BatchNorm1d(256)
        self.linear3 = Linear(256, 128)
        self.linear3_batchnorm = BatchNorm1d(128)
        self.linear4 = Linear(128, output)
        self.linear4_activation = config["output_ac"]

    def forward(self, X):
        """
        Going through the Layers and predicting
        """
        X = X.view(-1, 3, 84, 84)
        preds = self.activation(
            self.max_pool2d(self.dropout_conv(self.conv1_batchnorm(self.conv1(X))))
        )
        preds = self.activation(
            self.max_pool2d(self.dropout_conv(self.conv2_batchnorm(self.conv2(preds))))
        )
        preds = preds.view(-1, 16 * 18 * 18)
        preds = self.activation(
            self.dropout_linear(self.linear1_batchnorm(self.linear1(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear2_batchnorm(self.linear2(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear3_batchnorm(self.linear3(preds)))
        )
        preds = self.linear4_activation(self.linear4(preds))
        return preds
