# Prediction of Color RGB values from words
Mini-project for L101 course work.
Pbjection is;
- Is it possible to capture RGB related concept with Word2Vec trained with only sentences with color word
    - With static embedding like Word2Vec
    - With dynamic embedding like glove
- Is it possible to predict RGB values from words
    - with word embedding with sub-wirds static embedding
    - With dynamic embedding (wiht Liner transformations)
    - With RNN(Transformer) + FNN (with context)

# Color words extraction
The color words list is extracted from the website;
- https://spokenenglishtips.com/colors-name-in-english/
- https://www.facemediagroup.co.uk/resources/a-to-z-guides/a-to-z-of-colours

Script uesed to extract the color and hex values is color_list_scrapeing.py.

# Corpus filtering
The sentences with color words are filtered.
The corpus used is;
- NLTK.gutenberg

Script uesed to filter the sentences is color_data_extraction.py.

# Static Model
gensim word2vec with word embedding size 3:  
In conclusion, the model did not captured the RGB or color aspect of words.  
In normal word2vec simirality and relation representaion, it failed to express the colos.  
By extracting color word2vecs from the model and compared it with original RGB with;
- Nearest color analysis -> Failed
- Adding color relations -> Failed
- Adding color nearest -> Failed

## 3 dimension embedding (Base model)
> with 3 dim:  
The color (adding) relation agreements rate:  
0.014919011082693947  
The nearest colors overlapping more than 2 agreements rate:  
0.2608695652173913  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.06692242114236999

> with 3dim converted to 3dim:  
The color (adding) relation agreements rate:  
0.026854219948849106  
The nearest colors overlapping more than 2 agreements rate:  
0.2898550724637681  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.07800511508951406  

## 100 dimension embedding
> with 100 dim:  
The color (adding) relation agreements rate:  
0.060528559249786874  
The nearest colors overlapping more than 2 agreements rate:  
0.2898550724637681  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.13640238704177324

> with 100dim converted to 3dim:  
The color (adding) relation agreements rate:  
0.02216538789428815  
The nearest colors overlapping more than 2 agreements rate:  
0.30434782608695654  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.0886615515771526

However, sometimes it is reasonable for nearest(red,word2vec), so it looks model partially captured the concept.
>**Gold standard RGB vectors**  
nearest("red",rgbs01)  
Most Similar Color:  red  
0.9408929189489255 hazel  
0.9610744623273785 crimson  
0.9901811340592592 scarlet  
1.0 red  
1.0 maroon  
['hazel', 'crimson', 'scarlet', 'red', 'maroon']

> **3dim embedding vector**  
>nearest("red",rgbs02)  
Most Similar Color:  red  
0.9976268811067762 auburn  
0.9997738830062823 tan  
0.9999999999999999 red  
0.9982094721950749 white  
0.9984172522632984 coral  
['auburn', 'tan', 'red', 'white', 'coral']

> **3dim to 3dim Fnn vector**  
> nearest("red",rgbs02)  
Most Similar Color:  red  
0.9985647106567173 plum  
0.9995697476609071 pink  
0.9997614300433801 ruby  
0.9999999999999999 red  
0.9998100043611121 russet  
['plum', 'pink', 'ruby', 'red', 'russet']

# Dinamic Model
- https://github.com/huggingface/transformers
- https://note.com/npaka/n/n5bb043191cc9
- https://note.com/npaka/n/n49bec37cdd1b
- https://zenn.dev/ttya16/articles/3c51001f9e6d4b0ecc0b

**Bert embedding model with 768 dims:**
>The color (adding) relation agreements rate:  
0.014264936227343926  
The nearest colors overlapping more than 2 agreements rate:  
0.036585365853658534  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.035149548743193856

**RGB-FNN in BERT embedding:**
>The color (adding) relation agreements rate:  
0.0032818676810621315  
The nearest colors overlapping more than 2 agreements rate:  
0.04573170731707317  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.0036921011411948984

**BERT+RGB_FNN**
>The color (adding) relation agreements rate:  
0.0007272320429626315  
The nearest colors overlapping more than 2 agreements rate:  
0.0  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
1.8646975460580295e-05

## Normalization re-scaling
- https://atmarkit.itmedia.co.jp/ait/articles/2110/07/news027.html  

**FNN-RGB normalization:**  
Z-score -> subtract mean -> min-max -> round(x*255)
>The color (adding) relation agreements rate:  
0.0007085850675020511  
The nearest colors overlapping more than 2 agreements rate:  
0.003048780487804878  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.00026105765644812413

**BERT Embedding RGB normalization:**  
Z-score -> min-max -> round(x*255)
>The color (adding) relation agreements rate:  
0.002647870515402402  
The nearest colors overlapping more than 2 agreements rate:  
0.06097560975609756  
The color (adding) relation nearest overlap (more than 2) agreements rate:  
0.003953158797643022

# Color Plot
To be honest, all of the model failed.
But *FNN_in_BERT_RGB_Norm* model at least succeded to capture the shade of the colors.

# Concept mapping
*Concept_FNN_in_BERT_norm* looks the best, since it suggested suitable and consistent nearest colors to the theme.

# Reference
- https://qiita.com/sentencebird/items/fd0a9aae4c21f0753948
- https://nazology.net/archives/117062
- http://www.sakamoto-lab.hc.uec.ac.jp/research/
