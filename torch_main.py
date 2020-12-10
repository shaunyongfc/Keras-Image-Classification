import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import *
import image_loader
import torch_model

class ImageClassMain():
    def __init__(self):
        device = torch.device('cpu')
        net = torch_model.ImageClassTorch().to(device)
        opt = optim.Adam(
            net.parameters(),lr=INIT_LR,weight_decay=INIT_LR / EPOCHS)
        celoss = nn.CrossEntropyLoss()
        X_train, y_train = image_loader.get_images_train()
        trainset = []
        for i, j in enumerate(y_train):
            trainset.append([X_train[i], j])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BS)
        X_val, y_val = image_loader.get_images_test()
        valset = []
        for i, j in enumerate(y_val):
            valset.append([X_val[i], j])
        val_loader = torch.utils.data.DataLoader(valset, batch_size=BS)
        for epoch in range(EPOCHS):
            net.train()
            running_loss = 0
            for i, (data, target) in enumerate(train_loader):
                print(data.shape)
                print(target.shape)
                opt.zero_grad()
                output = net(data.to(device))
                loss = celoss(output, target.to(device))
                loss.backward()
                opt.step()
                running_loss += loss.item()
                if i % BS == (BS - 1):
                    print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                          % (epoch + 1, i + 1, runningLoss / BS))
                    runningLoss = 0
                net.eval()
                test_loss = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = net(data.to(device))
                        test_loss += celoss(output, target.to(device)).item()
                    print("Epoch: %2d, Validation Loss: %.3f"
                          % (epoch + 1, test_loss / BS))
        self.net = net
    def image_predict(self, number):
        """
        Predict a category from a numbered file in the pred folder.
        """
        X_pred = image_loader.get_pred_image(number)
        #y_pred = int(np.argmax(self.model.predict(X_pred), axis=-1))
        #return CATEGORIES[y_pred]

if __name__ == '__main__':
    image_class = ImageClassMain()
