import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import re
from decimal import Decimal

def str2rgb(s):
    nums = re.findall(r"\d+.\d+",s)
    return [int(float(n)) for n in nums]

def str2list(s):
    nums = re.findall(r"-?\d+\.\d+",s)
    return [float(n) for n in nums]

# Min-Max normalization
def min_max(vecs,val=None):
    val = vecs if val==None else val
    scaler = MinMaxScaler()
    scaler.fit(vecs)
    new_vecs = scaler.transform(val)
    return new_vecs

# Z-score normalization
def zscore_rgb(vecs):
    vec1, vec2, vec3 = vecs[:,0], vecs[:,1], vecs[:,2]
    zvec1, zvec2, zvec3 = stats.zscore(vec1), stats.zscore(vec2), stats.zscore(vec3)
    return np.array([[v1,v2,v3] for (v1,v2,v3) in zip(zvec1, zvec2, zvec3)])

def mean_sub(vecs):
    vec1, vec2, vec3 = vecs[:,0], vecs[:,1], vecs[:,2]
    m1, m2, m3 = np.mean(vec1), np.mean(vec2), np.mean(vec3)
    m_vec1, m_vec2, m_vec3 = vec1-m1, vec2-m2, vec3-m3
    return np.array([[v1,v2,v3] for (v1,v2,v3) in zip(m_vec1, m_vec2, m_vec3)])


if __name__ == '__main__':
    with open("original/color_100_fnn.txt","r",encoding="utf-8") as f:
        color_list = f.readlines()
    
    color_list = [c for c in color_list if c]
    colors = [c.split(":")[0] for c in color_list]
    rgbs = [np.array(str2rgb(c.split(":")[1].split("|")[0])) for c in color_list]
    vecs = np.array([np.array(str2list(c.split(":")[1].split("|")[1])) for c in color_list])
    
    new_vecs = zscore_rgb(vecs)
    #new_vecs2 = mean_sub(new_vecs)
    new_vecs3 = min_max(new_vecs)
    for i in range(len(vecs)):
        print(vecs[i],new_vecs3[i])
    new_rgbs = np.array([np.round(rgb).tolist() for rgb in new_vecs3*255])
    
    with open("normalized/color_100_fnn_norm.txt","w",encoding="utf-8") as f:
        for c,v1,v2 in zip(colors,rgbs,new_rgbs):
            f.write(c+":"+str(tuple(v1))+"|"+str(tuple(v2))+"\n")


