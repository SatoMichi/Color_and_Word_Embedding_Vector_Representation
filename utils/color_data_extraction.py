import sys
import nltk
from nltk.corpus import gutenberg

nltk.download('gutenberg')

with open(sys.path[0] + "/../data/color_list.txt", 'r') as f:
    txt = f.read()
colors = txt.split("\n")
colors = [c.split(":")[0] for c in colors]

if __name__ == "__main__":
    for color in colors:
        print(color)
        with open(sys.path[0] + "/../data/corpus/"+str(color)+".txt", "w", encoding='utf-8') as f:
            color = [color] if len(color.split(" "))==1 else [color,"-".join(color.split(" "))]
            for book in gutenberg.fileids():
                for sent in gutenberg.sents(book):
                    if any([c in [w.lower() for w in sent] for c in color]):
                        print(sent)
                        f.write(" ".join(sent)+"\n")
                    else: continue
                    

