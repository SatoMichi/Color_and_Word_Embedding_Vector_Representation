# Color information in Word Embedding Vector
This is public repository for mini-research project in L101 Machine Learning for Language Processing course.

## Introduction
**This research aimed to determine whether recent word embedding vectors generated from static or dynamic word embedding models could obtain a visual representation of the color words**. Word-embedded vectors are known as their arithmetic operation, which enables representing the semantic relations between the objects of the words being embedded. Since color is a concept that has a strong relative relationship with each other and could be learned from the context of the words, we came up with the idea use of word embedding to obtain a visual representation of the colors. In this research, we first focused on the possibility of the word embedding model to obtain the color representation like RGB, then tried to apply this model to predict the visual representation (RGB) of the non-color concept or unseen words.

## Word embedding vector representation and Color
The vector representation of word embedding can deal with some semantics of the words.  
$Vector(King) − Vector(man) + Vector(woman) ≈ Vector(Queen)$  

However, when we tried some different queries to word vector we observe the following.  
$Vector(Zebra) − Vector(stripe) + Vector(Brown) ≈ Vector(Giraffe)$ 
expected “Horse”   
$Vector(BlueBerry) − Vector(Blue) + Vector(Red) ≈ Vector(Cherry)$
expected “Strawberry”  

From these observations, it became clear that the word embedding vector is actually able to capture the visual features of the object partially. However, compared to our expectations, it is reasonable to say that **the vector did not completely encode color information**. We assumed this is caused by too much information other than color words included during the training. So we hypothesized that if we can train or fine-tune the word embedding model with only sentences with color words included, the resultant vector should capture color representation more clearly. 

Even more, we expected these vectors could capture the physical wavelength information about the color (for example RGB values, etc.) so that calculations like;  
$Vector(Yellow) + Vector(Blue) ≈ Vector(Green)$

## Code
For the details of the experiments conducted, please refer to the *L101_report.pdf*. In this section, we will briefly explain which code are used for whcih experiments.

### Data
For the training data or data for fine-tuning, we used sentences from the Gutenberg corpus included in NLTK python packages. The script used to extract the useful sentences form the corpus is *color_data_extraction.py* in *utils*.

We also extracted the color-word information from following websites and saved these data into *data* directory.
- https://spokenenglishtips.com/colors-name-in-english/
- https://www.facemediagroup.co.uk/resources/a-to-z-guides/a-to-z-of-colours

Script uesed to extract the color and hex values is *color_list_scrapeing.py* in *utils*.

### Experiment01: word2vec model
The directory: *src01*.  
This experiments is about observing the **word2vec** representation of the color words. If you want to know what is word2vec please look at this [Wikipedia page](https://en.wikipedia.org/wiki/Word2vec). The word2vec model are imported from gensim library. We trained or fine-tuned word2vec model and evaluated whether the color word vector contains the visual information.  

We built the following models;
- Word2Vec with 3dim vector representation + Feedforward Neural Network (FNN) with output size 3
- Word2Vec with 100dim vector representation + Feedforward Neural Network (FNN) with output size 3  

The output size is limited to 3 because we want to compare it with RGB representation of the color.

### Experiment02: BERT momdel
The directory: *src02*.  
This experiments is about observing the **BERT** word embedding representation of the color words. If you want to know what is BERT please look at this [Wikipedia page](https://en.wikipedia.org/wiki/BERT_(language_model)). The BERT model are imported from [Hugggingface Transformers](https://huggingface.co/docs/transformers/index). We trained or fine-tuned BERT model and evaluated whether the color word vector contains the visual information.

We built the following models;
- BERT embedding model with 768dim vector representation + Feedforward Neural Network (FNN) with output size 3  

The output size is limited to 3 because we want to compare it with RGB representation of the color.

## Experiments
The process of the experiments is simple. 

We first trained or fine-tuned the word embedding model with our data. 

Then we conducted some arithmetic operation between color vectors obtained from these models. The operation is designed to evaluate the visual information of the word embedding representation.
1. Evaluation metric01: Color vector adding   
$Vector(Yellow) + Vector(Blue) ≈ Vector(Green)$, then got point.
2. Evaluation metric02: Color vector similarity  
**{ color for vector in 5 nearest vectors of Vector(Yellow)}** overlaps with actual colors similar to Yellow, then got point.
3. Evaluation metric03: Color vector adding similarity  
**{ color for vector in 5 nearest vectors of Vector(Yellow)+Vector(Blue)}** overlaps with actual colors similar to Yellow+Blue, then got point.

※ At this point we applied some normarization process on the obtained vector data.

Finally for fun, we tried to generate color from words;
1. Obtain color vector Vector(*word*).
2. Compute the color representation by calculating weighted average of 5 nearest color vector for Vector(*word*).

![fruit](https://github.com/SatoMichi/Color_and_Word_Embedding_Vector_Representation/assets/44910734/053445fc-3c51-4a4d-9d17-4aafd92bf02b)

## Conclusion
By comparing the experiment results of word2vector with the BERT-based one which is pre-trained, it is reasonable to conclude **it is possible for the word embedding model to capture the color features of the words** if proper training data is given. 

This representation is able to capture the similarity between the color concept, however, it is not sufficient for expressing the physical property of colors like color addition (which means the representation did not capture the RGB-like features inducted from the physical property of the colors). This observation leads to the conclusion that **the task of capturing the physical property of the color (like RGB) is much more difficult than finding out the similarity of colors** since the word embedding model only learns the representation from distributed contexts. 
