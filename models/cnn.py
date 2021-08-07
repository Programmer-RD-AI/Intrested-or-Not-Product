from imports import *

"""
Using Conv2d instend of Conv1d
"""

class CNN(Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Intialization
        """
        output = config["output"]
        self.activation = LeakyReLU()
        self.dropout_conv = Dropout()
        self.max_pool2d = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = Conv2d(3, 8, (5, 5))
        self.conv1_batch_norm = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 16, (5, 5))
        self.conv2_batch_norm = BatchNorm2d(16)
        self.conv3 = Conv2d(16, 32, (5, 5))
        self.conv3_batch_norm = BatchNorm2d(32)
        self.dropout_linear = Dropout()
        self.linear1 = Linear(32 * 7 * 7, 512)
        self.linear1_batch_norm = BatchNorm1d(512)
        self.linear2 = Linear(512, 1024)
        self.linear2_batch_norm = BatchNorm1d(1024)
        self.linear3 = Linear(1024, 2048)
        self.linear3_batch_norm = BatchNorm1d(2048)
        self.linear4 = Linear(2048, 1024)
        self.linear4_batch_norm = BatchNorm1d(1024)
        self.linear5 = Linear(1024, output)
        self.linear5_activation = config["output_ac"]

    def forward(self, X):
        """
        Going through the Layers and predicting
        """
        X = X.view(-1, 3, 84, 84)
        preds = self.activation(
            self.dropout_conv(self.max_pool2d(self.conv1_batch_norm(self.conv1(X))))
        )
        preds = self.activation(
            self.dropout_conv(self.max_pool2d(self.conv2_batch_norm(self.conv2(preds))))
        )
        preds = self.activation(
            self.dropout_conv(self.max_pool2d(self.conv3_batch_norm(self.conv3(preds))))
        )
        preds = preds.view(-1, 32 * 7 * 7)
        preds = self.activation(
            self.dropout_linear(self.linear1_batch_norm(self.linear1(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear2_batch_norm(self.linear2(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear3_batch_norm(self.linear3(preds)))
        )
        preds = self.activation(
            self.dropout_linear(self.linear4_batch_norm(self.linear4(preds)))
        )
        preds = self.linear5_activation(self.linear5(preds))
        return preds
