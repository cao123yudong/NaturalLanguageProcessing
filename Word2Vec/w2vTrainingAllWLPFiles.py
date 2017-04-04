#Word2Vec model on text files of a small 25-file subset of sample COCA corpus
#codes modified from the tutorial https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb

# import modules & set up logging, may not use all of them, just in case
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import cython
from gensim import corpora, models, similarities
import gensim
from sklearn import svm, metrics
import numpy as np
import smart_open, os

import string
alphabet = string.ascii_lowercase


# set up MySentences class to import lemmatizations for training W2V model
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fcorpora = []
            for line in open(os.path.join(self.dirname, fname), encoding = "ISO-8859-1"): #to fix UnicodeDecodeError
                if len(line.split()) == 3:
                    if 'corrupt' in line.split()[1]: #refine lemmatization of corrupt
                        fcorpora.append('corrupt')
                    else:
                        if any(i in line.split()[1] for i in alphabet): #leave out special char & #s
                            fcorpora.append(line.split()[1])
            yield fcorpora



# trial1 for 115 WLP txt files for the COCA sample data
sentences1 = MySentences('/Users/yudongcao/PycharmProjects/NLP/wordLemPos')
len(list(sentences1))
list(sentences1)[1][:10:]
len(list(sentences1)[1])#15069


# train W2V model
model1 = gensim.models.Word2Vec(sentences1, min_count=5, size=200, workers=10)
print(model1)
model1.vocab


# finished modeling, trim unneeded model memory
model1.init_sims(replace=True)

# find similarity between words, for example
# model1.similarity('corrupt', 'corruption')

# top 10 most similar words
model1.most_similar(positive=['corrupt'])



# My Codes for:
# calculate similarities
simtocorrupt1 = []
for wordlst in iter(list(sentences1)):
    for word in iter(wordlst):
        try:
            sim = model1.similarity('corrupt', word)
        except KeyError:
                sim = 0
        simtocorrupt1.append([sim, word])

#sort similarities from high to low
ss = sorted(simtocorrupt1)[::-1]

#produce unique records
sorted1 = [ss[0]]
for i in range(1, len(ss)):
    if ss[i] != ss[i-1]:
        sorted1.append(ss[i])

# first 50 most similar words to 'corrupt'
sorted1[:50:]


# Out[175]: model1 = gensim.models.Word2Vec(sentences1, min_count=5, size=200, workers=10)

[[1.0, 'corrupt'],
 [0.96380239084427533, 'ideology'],
 [0.96138741540321282, 'destiny'],
 [0.95911861523579867, 'capable'],
 [0.95674791991156416, 'inhibit'],
 [0.95542007110106097, 'strict'],
 [0.95466179238900151, 'integrity'],
 [0.9538967339195884, 'works'],
 [0.95300671020547623, 'transformation'],
 [0.95299861251052209, 'intolerance'],
 [0.9519440402417616, 'intention'],
 [0.95131105081069911, 'frequent'],
 [0.95093376025185838, 'essence'],
 [0.95063300545723339, 'narrative'],
 [0.94999919217502304, 'prosecute'],
 [0.9482878425645469, 'disagreement'],
 [0.94818549184423251, 'effectiveness'],
 [0.9480173606319594, 'cosmos'],
 [0.94621242528526728, 'ego'],
 [0.94507165903177948, 'c-fern'],
 [0.94501814096193915, 'blau'],
 [0.94465364509515237, 'underestimate'],
 [0.94428054626309255, 'bias'],
 [0.94420425722057744, 'suspicious'],
 [0.94376998602110418, 'tenured'],
 [0.94369333416583123, 'feasible'],
 [0.94337850203172302, 'analyze'],
 [0.94330506779113676, 'pluralism'],
 [0.94305752840879054, 'availability'],
 [0.94302613904885746, 'hence'],
 [0.94255891002724379, 'pluralistic'],
 [0.94220327285571481, 'scholarship'],
 [0.94217138806570777, 'motive'],
 [0.94189378024449111, 'psyche'],
 [0.94154377828291147, 'appetite'],
 [0.94133242553749796, 'pure'],
 [0.94084937261889701, 'ecology'],
 [0.94083519713334873, 'total-return'],
 [0.94072013667131849, 'principally'],
 [0.94064480249365812, 'beneficiary'],
 [0.94042357839634805, 'orthodox'],
 [0.94015145513810583, 'indigenous'],
 [0.94012358809675778, 'worldview'],
 [0.94000737621957131, 'adopt'],
 [0.93990520791298704, 'governance'],
 [0.93986466633228583, 'illiberal'],
 [0.93984277687659634, 'im'],
 [0.9398048288159655, 'logic'],
 [0.93978212131639127, 'imaginaries'],
 [0.93927180891756012, 'discourse']]
