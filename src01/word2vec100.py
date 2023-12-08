from gensim.models import word2vec
import os
import sys


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

with open(sys.path[0] + "/../data/color_list.txt", 'r') as f:
    txt = f.read()
color_list = txt.split("\n")
color_list = [c for c in color_list if c]
colors = [c.split(":")[0] for c in color_list]
rgbs = [hex_to_rgb(c.split(":")[1]) for c in color_list]
color = dict(zip(colors, rgbs))

def get_wv(word):
    return model.wv[word]

def add_wv(w1,w2):
    return model.wv.most_similar(positive=[w1, w2], negative=[], topn=5)

sentences = []
for txt in os.listdir(sys.path[0] + "/../data/corpus/"):
    with open(sys.path[0] + "/../data/corpus/"+txt, "r", encoding="utf-8") as f:
        sentences += f.readlines()
sentences = [s.split(" ") for s in sentences if s]
print(len(sentences))

model = word2vec.Word2Vec(sentences, vector_size=100, sg=1)
model.save("model/word2vec-100.model")

with open("color_vec/lists/color_100_word2vec.txt", "w", encoding="utf-8") as f:
    for c,v in color.items():
        try:
            wv = get_wv(c)
            f.write(c+":"+str(v)+"|"+str(tuple(wv))+"\n")
        except:
            pass

        
    