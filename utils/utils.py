import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def rgb_to_hex(rgb):
    if len(rgb) > 3: rgb = rgb[1:]
    return '#%02x%02x%02x' % rgb
    
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def plot_tsne(x, y, color):
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.title("Color relationship with embedding vectors")
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    x_embedded = tsne.fit_transform(x)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=color)
    plt.show()