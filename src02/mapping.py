import numpy as np
import torch
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import os
import sys
from tqdm import tqdm

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

with open(sys.path[0] + "/../data/color_list.txt", 'r') as f:
    txt = f.read()
color_list = txt.split("\n")
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
hexs = [c.split(":")[1] for c in color_list]
color = dict(zip(colors, hexs))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("model")

def str2rgb(s):
    nums = re.findall(r"\d+",s)
    return [float(n) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d\.\d+",s)
    return [float(n) for n in nums]

with open("color_vec/lists/color_embbedding.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
hexs = [c.split(":")[1].split("|")[0] for c in color_list]
rgbs = np.array([np.array(str2rgb(c.split(":")[1].split("|")[1].split("$")[0])) for c in color_list])
vecs = [c.split(":")[1].split("|")[1].split("$")[1] for c in color_list]
vecs = [np.array(str2list(v)) for v in vecs]

def encode_texts(texts):
    encoding = tokenizer.batch_encode_plus(texts, return_tensors='pt', pad_to_max_length=True)
    return encoding['input_ids'], encoding['attention_mask']

def plot_tsne(x, y, color):
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.title("Color relationship with embedding vectors")
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    x_embedded = tsne.fit_transform(x)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=color)
    plt.show()

if __name__ == "__main__":
    # すべてのパラメータを固定
    for param in model.parameters():
        param.requires_grad = False
    # change last layer to RGB mapper
    n_in_features = model.pooler.dense.in_features
    n_out_features = 3
    model.pooler.dense = torch.nn.Linear(n_in_features, n_out_features)
    model.pooler.activation = torch.nn.ReLU()


    # prepare data
    ids, attention_mask = encode_texts(colors)

    # train last layer
    y_tensor = torch.from_numpy(rgbs).float().to("cuda")
    i_train, i_val = train_test_split(range(len(ids)), test_size=0.20)
    print(y_tensor[i_train].shape)
    print(ids[i_val].shape)

    # start training
    optimizer = transformers.AdamW(model.pooler.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    model.train()
    model.to("cuda")

    n_epochs = 5000
    train_losses, val_losses = [], []

    x_train, x_val, y_train, y_val = ids[i_train], ids[i_val], y_tensor[i_train], y_tensor[i_val]
    attention_mask_train, attention_mask_val = attention_mask[i_train], attention_mask[i_val]
    x_train = x_train.to("cuda")
    x_val = x_val.to("cuda")
    y_train = y_train.to("cuda")
    y_val = y_val.to("cuda")
    attention_mask_train =  attention_mask_train.to("cuda")
    attention_mask_val = attention_mask_val.to("cuda")
    
    for epoch in tqdm(range(n_epochs)):
        y_train_pred = model(x_train, attention_mask_train)
        loss = criterion(y_train_pred.pooler_output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.data.to("cpu").numpy().tolist())

        y_val_pred = model(x_val, attention_mask_val)
        val_loss = criterion(y_val_pred.pooler_output, y_val)
        val_losses.append(val_loss.data.to("cpu").numpy().tolist())

    model.save_pretrained("mapped_model")

    plt.figure()
    plt.title("Train and Test Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1,n_epochs+1),train_losses,color="b",linestyle="-",label="train_loss")
    plt.plot(range(1,n_epochs+1),val_losses,color="r",linestyle="-",label="test_loss")
    plt.legend()
    plt.show() 

    # Evaluation of the color word
    texts = list(color.keys())
    ids, attention_mask = encode_texts(texts)
    model.to("cpu")
    outputs = model(ids)
    vecs = np.array(outputs[0][:, 0, :].tolist())

    with open("color_vec/lists/color_emb_rgb.txt", "w", encoding="utf-8") as f:
        for i,(c,v) in enumerate(color.items()):
            wv = vecs[i]
            f.write(c+":"+str(v)+"|"+str(hex_to_rgb(v))+"$"+str(tuple(wv))+"\n")

    plot_tsne(vecs, texts, color.values())