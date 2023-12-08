import re
import numpy as np
from itertools import combinations
import sys

def str2rgb(s):
    nums = re.findall(r"\d+",s)
    return [float(n) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d\.\d+",s)
    return [float(n) for n in nums]

with open("lists\color_100_fnn.txt","r",encoding="utf-8") as f:
    color_list = f.readlines()

color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
rgbs01 = [np.array(str2rgb(c.split(":")[1].split("|")[0])) for c in color_list]
rgbs02 = [np.array(str2list(c.split(":")[1].split("|")[1])) for c in color_list]

color_rgb = dict(zip(colors,rgbs01))
word2vec_rgb = dict(zip(colors,rgbs02))

def add_vec(v1,v2):
    return (v1+v2)/2

def cos_sim(v1, v2):
    v1 = v1+0.000000001
    v2 = v2+0.000000001
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cossim_list(vs,v1):
    cossim = []
    for v2 in vs:
        cossim.append(cos_sim(v1,v2))
    return cossim

def most_similar(v1,vs):
    cossim = cossim_list(vs,v1)
    i = np.argmax(cossim)
    max_sim = cossim[i]
    return i, max_sim


def color_add(color01, color02, log=True):
    v1 = rgbs01[colors.index(color01)]
    v2 = rgbs01[colors.index(color02)]
    w1 = rgbs02[colors.index(color01)]
    w2 = rgbs02[colors.index(color02)]
    v3 = add_vec(v1,v2)
    w3 = add_vec(w1,w2)
    vi,vs = most_similar(v3,rgbs01)
    wi,ws = most_similar(w3,rgbs02)
    if log: print(color01,"+",color02,"=","RGB:",colors[vi],"W2V:",colors[wi])
    return True if vi==wi else False

def nearest(color,rgbs,n=-5,log=True):
    cossim = cossim_list(rgbs,rgbs[colors.index(color)])
    inds = np.argpartition(cossim,n)[n:]
    sims = np.array(cossim)[inds]
    if log: print("Most Similar Color: ",color)
    col = []
    for i in range(len(inds)):
        if log: print(sims[i],colors[inds[i]])
        col.append(colors[inds[i]])
    return col

def nearest_vec(vec,rgbs,n=-5,log=True):
    cossim = cossim_list(rgbs,vec)
    inds = np.argpartition(cossim,n)[n:]
    sims = np.array(cossim)[inds]
    col = []
    for i in range(len(inds)):
        if log: print(sims[i],colors[inds[i]])
        col.append(colors[inds[i]])
    return col 

def overlap(x,y,n=2):
    return True if len(set(x).intersection(set(y)))>=2 else False

def color_add_nearest(color01,color02,n=-5,log=True):
    v1 = rgbs01[colors.index(color01)]
    v2 = rgbs01[colors.index(color02)]
    w1 = rgbs02[colors.index(color01)]
    w2 = rgbs02[colors.index(color02)]
    v3 = add_vec(v1,v2)
    w3 = add_vec(w1,w2)
    if log: print(color01,"+",color02,"=")
    col1 = nearest_vec(v3,rgbs01,log=log)
    if log: print("RGB: ",col1)
    col2 = nearest_vec(w3,rgbs02,log=log)
    if log: print("W2V: ",col2)
    return overlap(col1,col2)

if __name__ == "__main__":
    print("The color (adding) relation agreements rate:")
    comb = combinations(colors, 2)
    result = [color_add(c1,c2,log=False) for c1,c2 in list(comb)]
    acc = sum(result)/len(result)
    print(acc)

    print("The nearest colors overlapping more than 2 agreements rate:")
    result = [overlap(nearest(c,rgbs01,log=False),nearest(c,rgbs02,log=False)) for c in colors]
    acc = sum(result)/len(result)
    print(acc)

    print("The color (adding) relation nearest overlap (more than 2) agreements rate:")
    comb = combinations(colors, 2)
    result = [color_add_nearest(c1,c2,log=False) for c1,c2 in list(comb)]
    acc = sum(result)/len(result)
    print(acc)
