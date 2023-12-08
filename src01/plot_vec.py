import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys
import re

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    if len(rgb) > 3: rgb = rgb[1:]
    return '#%02x%02x%02x' % rgb

def str2rgb(s):
    nums = re.findall(r"\d+.\d+",s)
    return [int(float(n)) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d\.\d+",s)
    return [float(n) for n in nums]

def plot_tsne(x, y, color):
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.title("Color relationship with embedding vectors")
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    x_embedded = tsne.fit_transform(x)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=color)
    plt.show()


with open(sys.path[0] + "/../data/color_list.txt", 'r') as f:
    txt = f.read()
color_list = txt.split("\n")
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
hexs = [c.split(":")[1] for c in color_list]
color2hex = dict(zip(colors, hexs))

file = os.listdir("color_vec/lists/")
for f in file:
    print(f)
    with open("color_vec/lists/"+f,"r",encoding="utf-8") as f:
        color_list = f.readlines()
    color_list = [c for c in color_list if c]
    colors = [c.split(":")[0] for c in color_list]
    hexs = [color2hex[c] for c in colors]
    rgbs02 = np.array([np.array(str2list(c.split(":")[1].split("|")[1])) for c in color_list])
    plot_tsne(rgbs02, [], hexs)
