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

with open("color_emb_rgb_norm.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
hexs = [rgb_to_hex(tuple(str2rgb(c.split(":")[1].split("|")[0]))) for c in color_list]
vecs = np.array([str2list(c.split(":")[1].split("|")[1]) for c in color_list])
c2h = {c:h for c,h in zip(colors,hexs)}
c2v = {c:v for c,v in zip(colors,vecs)}


with open("concept_only_emb_rgb_norm.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()
color_list = [c for c in color_list if c]
concepts = [c.split(":")[0] for c in color_list]
con_vecs = [str2list(c.split(":")[1]) for c in color_list]

def cos_sim(v1, v2):
    v1 = v1+0.000000001
    v2 = v2+0.000000001
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cossim_list(vs,v1):
    cossim = []
    for v2 in vs:
        cossim.append(cos_sim(v1,v2))
    return cossim

def nearest(vec,rgbs,n=-5,log=True):
    cossim = cossim_list(rgbs,vec)
    inds = np.argpartition(cossim,n)[n:]
    sims = np.array(cossim)[inds]
    if log: print("Most Similar Color: ",vec)
    col = []
    for i in range(len(inds)):
        if log: print(sims[i],colors[inds[i]])
        col.append(colors[inds[i]])
    return col

for i in range(len(concepts)):
    concept = concepts[i]
    rgb = np.array(con_vecs[i])
    nearest_colors = nearest(rgb,vecs,log=False)
    nearest_hexs = [c2h[c] for c in nearest_colors]
    hex2 = rgb_to_hex(tuple(rgb))
    print(concept, hex2, nearest_colors)
    
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(1,6,1)
    ax1.set_title("\""+concept+"\" color Predicted:")
    ax1.set_facecolor(hex2)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    for j in range(5):
        ax_sub = plt.subplot(1,6,j+2)
        ax_sub.set_facecolor(nearest_hexs[j])
        if j==0: ax_sub.set_title("Nearest colors")
        ax_sub.axes.xaxis.set_visible(False)
        ax_sub.axes.yaxis.set_visible(False)
    plt.savefig(concept+".png")
    
