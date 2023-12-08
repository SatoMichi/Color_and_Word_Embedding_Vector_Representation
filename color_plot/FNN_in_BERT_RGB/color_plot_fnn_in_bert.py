import re
import numpy as np
import matplotlib.pyplot as plt

def str2rgb(s):
    nums = re.findall(r"\d+",s)
    return [int(float(n)) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d+\.\d+",s)
    return [np.round(float(n)).astype(int) for n in nums]

def rgb_to_hex(rgb):
    if len(rgb) > 3: rgb = rgb[1:]
    return '#%02x%02x%02x' % rgb

with open("FNN_in_BERT_RGB/color_emb_rgb.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
hexs = [c.split(":")[1].split("|")[0] for c in color_list]
rgbs = np.array([np.array(str2rgb(c.split(":")[1].split("|")[1].split("$")[0])) for c in color_list])
vecs = [c.split(":")[1].split("|")[1].split("$")[1] for c in color_list]
vecs = [np.array(str2list(v)) for v in vecs]

for i in range(len(vecs)):
    color = colors[i]
    hex = hexs[i]
    rgb = vecs[i]
    hex2 = rgb_to_hex(tuple(rgb))
    print(color, hex, hex2)
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.set_title("Color : "+color)
    ax1.set_facecolor(hex)
    ax2.set_title("Predicted")
    ax2.set_facecolor(hex2)
    plt.savefig("FNN_in_BERT_RGB/"+color+".png")
