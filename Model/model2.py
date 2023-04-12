import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class MyModel(nn.Module):
    def __init__(self, file_path, args=None, num_classes=10):
        super(MyModel, self).__init__()
        if not os.path.exists(file_path):
            print("ERROR: File {} does not exist".format(file_path))
            sys.exit(1)
        df = pd.read_csv(file_path)  # dataframe
        feature_list = []
        max_flag=False
        average_flag=False
        classifier_flag=True
        for i, row in df.iterrows():
            # for each layer i
            if row['kernel height'] == 1 and row['kernel width'] == 1:
                in_features = row['IFM channel depth']
                out_features = row['kernel depth']
                linear = nn.Linear(in_features, out_features, bias=False)
                activation = None
                if row['activation'] == 'relu':
                    activation = nn.ReLU()
                elif row['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif row['activation'] == 'tanh':
                    activation = nn.Tanh()
                elif row['activation'] == 'softmax':
                    activation = nn.Softmax()

                if classifier_flag:  # meets a linear layer at the first time
                    self.features = nn.Sequential(*feature_list)
                    self.classifier = nn.Sequential(linear)
                    if activation:
                        self.classifier.append(activation)
                    classifier_flag = False
                else:
                    self.classifier.append(linear)
                    if activation:
                        self.classifier.append(activation)

            elif max_flag:
                kernel_size = (row['kernel height'], row['kernel width'])
                stride = row['stride']
                padding = row['padding']
                feature_list += [nn.MaxPool2d(kernel_size, stride, padding)]  # Max-pooling layer
                max_flag = False
            elif average_flag:
                kernel_size = (row['kernel height'], row['kernel width'])
                stride = row['stride']
                padding = row['padding']
                feature_list += [nn.AvgPool2d(kernel_size, stride, padding)]  # Average-pooling layer
                average_flag = False
            else:
                in_channel = row['IFM channel depth']
                out_channel = row['kernel depth']
                kernel_size = (row['kernel height'], row['kernel width'])
                stride = row['stride']
                padding = row['padding']
                conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
                activation = None
                if row['activation'] == 'relu':
                    activation = nn.ReLU()
                elif row['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif row['activation'] == 'tanh':
                    activation = nn.Tanh()
                elif row['activation'] == 'softmax':
                    activation = nn.Softmax()
                if activation:
                    feature_list += [conv2d, activation]  # Convolution layer with activation
                else:
                    feature_list += [conv2d]
                # pooling
                if row['max pooling'] == 1:
                    max_flag = True
                elif row['avg pooling'] == 1:
                    average_flag = True

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MyModel('../NetWork.csv')
    x = np.zeros([4,5])
    model.forward(x)

