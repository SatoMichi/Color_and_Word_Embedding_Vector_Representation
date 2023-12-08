import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
import re
import os
import sys

def str2rgb(s):
    nums = re.findall(r"\d+",s)
    return [float(n) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d\.\d+",s)
    return [float(n) for n in nums]

with open("lists/color_100_word2vec.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
rgb01 = np.array([np.array(str2rgb(c.split(":")[1].split("|")[0])) for c in color_list])
rgb02 = np.array([np.array(str2list(c.split(":")[1].split("|")[1])) for c in color_list])

# TensorDataset
train_torch = torch.from_numpy(rgb02).float()   # 100 dim
target_torch = torch.from_numpy(rgb01).float()   # 3 dim
train_set = TensorDataset(train_torch,target_torch)
train_batch = DataLoader(train_set, batch_size=5, shuffle=True,num_workers=0)

# FNN
class FNN100dim2RGB(torch.nn.Module):
    def __init__(self, input_size=100, hidden_size=30, output_size=3):
        super(FNN100dim2RGB,self).__init__()
        self.linear1 = torch.nn.Linear(input_size,hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size,hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    D_in = 100
    H = 30
    D_out = 3
    epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FNN100dim2RGB(D_in,H,D_out).to(device)

    if not os.path.exists("params/fnn100dim2rgb_net.pth"):
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.L1Loss()
        optimizer = optim.Adam(net.parameters())

        train_loss_list = []
        train_acc_list = []

        for i in range(epoch):
            print("##############################################################")
            print("Epoch: {}/{}".format(i+1,epoch))
            train_loss = 0
            train_acc = 0
            
            net.train()
            for data,label in train_batch:
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                y_pred = net(data)
                loss = criterion1(y_pred,label)
                mae = criterion2(y_pred,label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += mae.item()
            batch_train_loss = train_loss/len(train_batch)
            batch_train_acc = train_acc/len(train_batch)

            print("Train loss:{:.4f}, Train acc: {:.4f}".format(batch_train_loss,batch_train_acc))

            train_loss_list.append(batch_train_loss)
            train_acc_list.append(batch_train_acc)

        # visualization
        plt.figure()
        plt.title("Train and Test Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(1,epoch+1),train_loss_list,color="b",linestyle="-",label="train_loss")
        plt.legend()
        plt.figure()
        plt.title("Train and Test MAE")
        plt.xlabel("epoch")
        plt.ylabel("MAE")
        plt.plot(range(1,epoch+1),train_acc_list,color="b",linestyle="-",label="train_accuracy")
        plt.legend()
        plt.show()

        # save and load the model
        torch.save(net.to(device).state_dict(),"params/fnn100dim2rgb_net.pth")
    else:
        net.load_state_dict(torch.load("params/fnn100dim2rgb_net.pth"))
        net.eval()
        new_rgb = net(train_torch.to(device))
        new_rgb = new_rgb.tolist()

        with open("lists/color_100_fnn.txt","w",encoding="utf-8") as f: 
            for c,v1,v2 in zip(colors,rgb01,new_rgb):
                f.write(c+":"+str(tuple(v1))+"|"+str(tuple(v2))+"\n")


