import torch
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Pretrained Bert word embedding model 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # casedは大文字小文字区別なし
model = BertModel.from_pretrained('bert-base-cased')

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

if __name__ == '__main__':
    if not os.path.exists("model/config.json"):
        # Training (fine-tuning)
        sentences = []
        for txt in os.listdir(sys.path[0] + "/../data/corpus/"):
            with open(sys.path[0] + "/../data/corpus/"+txt, "r", encoding="utf-8") as f:
                sentences += f.readlines()
        sentences = [s for s in sentences if s]
        print(len(sentences))

        model.to("cuda")
        model.train()
        batch_size = 4
        optimizer = AdamW(model.parameters(), lr=1e-5) # AdamWオプティマイザ
        for i in range((len(sentences)//batch_size)+1):
            start_ind = i*batch_size
            end_ind = (i+1)*batch_size if (i+1)*batch_size<len(sentences) else len(sentences)-1
            sent = sentences[start_ind:end_ind]
            if sent == []: break
            print("Loop ",i,": ",len(sent)," sentences training start")
            encoding = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to("cuda")
            attention_mask = encoding['attention_mask'].to("cuda")
            labels = torch.tensor([1]*len(sent)).to("cuda")
            outputs = model(input_ids, attention_mask=attention_mask)
            #print(outputs)
            loss = F.cross_entropy(outputs.pooler_output, labels)
            loss.backward()
            optimizer.step()

        model.save_pretrained("model")
    else:
        model = BertModel.from_pretrained("model")

    # Evaluation of the color word
    texts = list(color.keys())
    ids, attention_mask = encode_texts(texts)
    outputs = model(ids)
    vecs = np.array(outputs[0][:, 0, :].tolist())

    with open("color_vec/lists/color_embbedding.txt", "w", encoding="utf-8") as f:
        for i,(c,v) in enumerate(color.items()):
            wv = vecs[i]
            f.write(c+":"+str(v)+"|"+str(hex_to_rgb(v))+"$"+str(tuple(wv))+"\n")

    plot_tsne(vecs, texts, color.values())
