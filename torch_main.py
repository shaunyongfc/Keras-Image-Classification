import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from parameters import *
import image_loader
import torch_model

# Define category names
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


class ImageClassMain():
    def __init__(self):
        """
        Main function that makes use of written functions in other files.
        """
        device = torch.device('cpu')
        # create model and define paremeters
        net = torch_model.ImageClassTorch().to(device)
        opt = optim.Adam(
            net.parameters(),lr=INIT_LR,weight_decay=INIT_LR / EPOCHS)
        celoss = nn.CrossEntropyLoss()
        # prepare data
        X_train, y_train = image_loader.get_images_train()
        X_train = X_train.transpose(0, 3, 1, 2)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        trainset = []
        for i, j in enumerate(y_train):
            trainset.append([X_train[i], j])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BS)
        X_val, y_val = image_loader.get_images_test()
        X_val = X_val.transpose(0, 3, 1, 2)
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).long()
        valset = []
        for i, j in enumerate(y_val):
            valset.append([X_val[i], j])
        val_loader = torch.utils.data.DataLoader(valset, batch_size=BS)
        # train model
        for epoch in range(EPOCHS):
            net.train()
            running_loss = 0
            for i, (data, target) in enumerate(train_loader):
                data, target = (data.to(device), target.to(device))
                opt.zero_grad()
                output = net(data)
                loss = celoss(output, target)
                loss.backward()
                opt.step()
                running_loss += loss.item()
                if i % 50 == 49:
                    print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                          % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0
            net.eval()
            test_loss = 0
            correct_pred = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = (data.to(device), target.to(device))
                    output = net(data)
                    output_pred = output.argmax(dim=1, keepdim=True)
                    correct_pred += output_pred.eq(target.view_as(output_pred)).sum().item()
                    test_loss += celoss(output, target).item()
                print("Epoch: %2d, Validation Loss: %.3f, Accuracy: %.3f"
                      % (epoch + 1, test_loss / len(val_loader), correct_pred / len(X_val)))
        self.net = net

    def image_predict(self, number):
        """
        Predict a category from a numbered file in the pred folder.
        """
        self.net.eval()
        X_pred = image_loader.get_pred_image(number)
        X_pred = X_pred.transpose(0, 3, 1, 2)
        X_pred = torch.from_numpy(X_pred).float()
        output = self.net(X_pred)
        y_pred = output.argmax(dim=1, keepdim=True)
        return CATEGORIES[y_pred]


if __name__ == '__main__':
    image_class = ImageClassMain()
