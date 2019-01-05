#!/usr/bin/env python
# coding: utf-8

# # Deep Learning for NLP - Project

# RULES:
# 
# * Do not create any additional cell
# 
# * Fill in the blanks
# 
# * All cells should be runnable (modulo trivial compatibility bugs that we'd fix)
# 
# * 4 / 20 points will be allocated to the clarity of your code
# 
# * Efficient code will have a bonus
# 
# DELIVERABLE:
# 
# * this notebook
# * the predictions of the SST test set
# 
# DO NOT INCLUDE THE DATASETS IN THE DELIVERABLE..

# In[1]:


import io
import os
import numpy as np
import scipy


# In[3]:


PATH_TO_DATA = "/data/" # On avait "../../data/" dans le fichier fourni


# # 1) Monolingual (English) word embeddings 

# In[128]:


class Word2vec():
    def __init__(self, fname, nmax=100000):
        self.load_wordvec(fname, nmax)
        self.word2id = dict.fromkeys(self.word2vec.keys())
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = np.array(self.word2vec.values())
    
    def load_wordvec(self, fname, nmax):
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break
        print('Loaded %s pretrained word vectors' % (len(self.word2vec)))

    def most_similar(self, w, K=5):
        # K most similar words: self.score  -  np.argsort
        similarWords = np.array([[key,self.score(w, key)]for key in self.word2vec.keys()])
        similarWords = similarWords[np.argsort(similarWords[:,1])]
        similarWords = similarWords[-K:,0]
        return(similarWords)

    def score(self, w1, w2):
        # cosine similarity: np.dot  -  np.linalg.norm
        vec1 = self.word2vec[w1]
        vec2 = self.word2vec[w2]
        score = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        return(score)


# In[106]:


w2v = Word2vec(os.path.join(PATH_TO_DATA, 'crawl-300d-200k.vec'), nmax=10000)
for w1 in ['cat']:
    print(w2v.most_similar(w1))


# In[117]:


print('cat', 'SOAR', w2v.score('cat', 'SOAR'))
print('cat', 'extensibility', w2v.score('cat', 'extensibility'))
print('cat', 'cats', w2v.score('cat', 'cats'))


# In[132]:


w2v = Word2vec(os.path.join(PATH_TO_DATA, 'crawl-300d-200k.vec'), nmax=25000)

# You will be evaluated on the output of the following:
for w1, w2 in zip(('cat', 'cat', 'cat', 'cat'), ('cats', 'Cat', 'dog', 'Acts')):
    print(w1, w2, w2v.score(w1, w2))
for w1 in ['cat', 'dog', 'dogs']:
    print(w2v.most_similar(w1))
'''for w1, w2 in zip(('cat', 'dog', 'dogs', 'paris', 'germany'), ('dog', 'pet', 'cats', 'france', 'berlin')):
    print(w1, w2, w2v.score(w1, w2))
for w1 in ['cat', 'dog', 'dogs', 'paris', 'germany']:
    print(w2v.most_similar(w1))'''


# In[7]:


class BoV():
    def __init__(self, w2v):
        self.w2v = w2v
    
    def encode(self, sentences, idf=False):
        # takes a list of sentences, outputs a numpy array of sentence embeddings
        # see TP1 for help
        sentemb = []
        return
        for sent in sentences:
            if idf is False:
                # mean of word vectors
                assert False, 'TODO: fill in the blank'
            else:
                # idf-weighted mean of word vectors
                assert False, 'TODO: fill in the blank'
        return np.vstack(sentemb)

    def most_similar(self, s, sentences, idf=False, K=5):
        # get most similar sentences and **print** them
        keys = self.encode(sentences, idf)
        query = self.encode([s], idf)
        return

    def score(self, s1, s2, idf=False):
        # cosine similarity: use   np.dot  and  np.linalg.norm
        return
    
    def build_idf(self, sentences):
        # build the idf dictionary: associate each word to its idf value
        idf = {}
        return
        # for sent in sentences:
        #     for w in set(sent):
        #         idf[w] = idf.get(w, 0) + 1
        
        # max(1, np.log10(len(sentences) / (idf[word])))


# In[8]:


w2v = Word2vec(os.path.join(PATH_TO_DATA, 'crawl-300d-200k.vec'), nmax=5000)
s2v = BoV(w2v)

# Load sentences in "PATH_TO_DATA/sentences.txt"
sentences = []

# Build idf scores for each word
idf = {} if True else s2v.build_idf(sentences)

# You will be evaluated on the output of the following:
s2v.most_similar('' if not sentences else sentences[10], sentences)  # BoV-mean
s2v.score('' if not sentences else sentences[7], '' if not sentences else sentences[13])


idf = {}  
s2v.most_similar('' if not sentences else sentences[10], sentences, idf)  # BoV-idf
s2v.score('' if not sentences else sentences[7], '' if not sentences else sentences[13], idf)


# # 2) Multilingual (English-French) word embeddings

# Let's consider a bilingual dictionary of size V_a (e.g French-English).
# 
# Let's define **X** and **Y** the **French** and **English** matrices.
# 
# They contain the embeddings associated to the words in the bilingual dictionary.
# 
# We want to find a **mapping W** that will project the source word space (e.g French) to the target word space (e.g English).
# 
# Procrustes : **W\* = argmin || W.X - Y ||  s.t  W^T.W = Id**
# has a closed form solution:
# **W = U.V^T  where  U.Sig.V^T = SVD(Y.X^T)**
# 
# In what follows, you are asked to: 

# In[1]:


# 1 - Download and load 50k first vectors of
#     https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
#     https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec

# TYPE CODE HERE


# In[2]:


# 2 - Get words that appear in both vocabs (= identical character strings)
#     Use it to create the matrix X and Y (of aligned embeddings for these words)

# TYPE CODE HERE


# In[3]:


# 3 - Solve the Procrustes using the scipy package and: scipy.linalg.svd() and get the optimal W
#     Now W*French_vector is in the same space as English_vector

# TYPE CODE HERE


# In[4]:


# 4 - After alignment with W, give examples of English nearest neighbors of some French words (and vice versa)
#     You will be evaluated on that part and the code above

# TYPE CODE HERE


# If you want to dive deeper on this subject: https://github.com/facebookresearch/MUSE

# # 3) Sentence classification with BoV and scikit-learn

# In[5]:


# 1 - Load train/dev/test of Stanford Sentiment TreeBank (SST)
#     (https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

# TYPE CODE HERE


# In[6]:


# 2 - Encode sentences with the BoV model above

# TYPE CODE HERE


# In[7]:


# 3 - Learn Logistic Regression on top of sentence embeddings using scikit-learn
#     (consider tuning the L2 regularization on the dev set)

# TYPE CODE HERE


# In[8]:


# 4 - Produce 2210 predictions for the test set (in the same order). One line = one prediction (=0,1,2,3,4).
#     Attach the output file "logreg_bov_y_test_sst.txt" to your deliverable.
#     You will be evaluated on the results of the test set.

# TYPE CODE HERE


# In[9]:


# BONUS!
# 5 - Try to improve performance with another classifier
#     Attach the output file "XXX_bov_y_test_sst.txt" to your deliverable (where XXX = the name of the classifier)

# TYPE CODE HERE


# # 4) Sentence classification with LSTMs in Keras

# ## 4.1 - Preprocessing

# In[10]:


import keras


# In[11]:


# 1 - Load train/dev/test sets of SST
PATH_TO_DATA = "../../data/"

# TYPE CODE HERE


# In[12]:


# 2 - Transform text to integers using keras.preprocessing.text.one_hot function
#     https://keras.io/preprocessing/text/

# TYPE CODE HERE


# **Padding input data**
# 
# Models in Keras (and elsewhere) take batches of sentences of the same length as input. It is because Deep Learning framework have been designed to handle well Tensors, which are particularly suited for fast computation on the GPU.
# 
# Since sentences have different sizes, we "pad" them. That is, we add dummy "padding" tokens so that they all have the same length.
# 
# The input to a Keras model thus has this size : (batchsize, maxseqlen) where maxseqlen is the maximum length of a sentence in the batch.

# In[13]:


# 3 - Pad your sequences using keras.preprocessing.sequence.pad_sequences
#     https://keras.io/preprocessing/sequence/

# TYPE CODE HERE


# ## 4.2 - Design and train your model

# In[14]:


# 4 - Design your encoder + classifier using keras.layers
#     In Keras, Torch and other deep learning framework, we create a "container" which is the Sequential() module.
#     Then we add components to this contained : the lookuptable, the LSTM, the classifier etc.
#     All of these components are contained in the Sequential() and are trained together.


# ADAPT CODE BELOW


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

embed_dim  = 32  # word embedding dimension
nhid       = 64  # number of hidden units in the LSTM
vocab_size = 0  # size of the vocabulary
n_classes  = 5

model = Sequential()
model.add(Embedding(vocab_size, embed_dim))
model.add(LSTM(nhid, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(n_classes, activation='sigmoid'))


# In[16]:


# 5 - Define your loss/optimizer/metrics

# MODIFY CODE BELOW

loss_classif     =  '' # find the right loss for multi-class classification
optimizer        =  '' # find the right optimizer
metrics_classif  =  ['accuracy']

# Observe how easy (but blackboxed) this is in Keras
model.compile(loss=loss_classif,
              optimizer=optimizer,
              metrics=metrics_classif)
print(model.summary())


# In[ ]:


# 6 - Train your model and find the best hyperparameters for your dev set
#     you will be evaluated on the quality of your predictions on the test set

# ADAPT CODE BELOW
bs = 64
n_epochs = 6

history = model.fit(x_train, y_train, batch_size=bs, nb_epoch=n_epochs, validation_data=(x_val, y_val))


# In[ ]:


# 7 - Generate your predictions on the test set using model.predict(x_test)
#     https://keras.io/models/model/
#     Log your predictions in a file (one line = one integer: 0,1,2,3,4)
#     Attach the output file "logreg_lstm_y_test_sst.txt" to your deliverable.

# TYPE CODE HERE


# ## 4.3 -- innovate !

# In[ ]:


# 8 - Open question: find a model that is better on your dev set
#     (e.g: use a 1D ConvNet, use a better classifier, pretrain your lookup tables ..)
#     you will get point if the results on the test set are better: be careful of not overfitting your dev set too much..
#     Attach the output file "XXX_XXX_y_test_sst.txt" to your deliverable.

# TYPE CODE HERE

