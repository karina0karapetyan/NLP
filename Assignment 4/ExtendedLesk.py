'''
Implement Extended Lesk algorithm explained in the book
'''

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize as w_tok
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
punctuation = ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '...', '\'', '\"', "\''", '``', '--', '-', '$']

RELS_ALL = ['gloss', 'examples', 'hyponyms', 'hypernyms', 'meronyms', 'holonyms', 'also-see', 'attribute']
RELS_NOUNS = ['gloss', 'examples', 'hyponyms', 'meronyms'] 
RELS_ADJS  = ['gloss', 'examples', 'also-see', 'attribute'] 
RELS_VERBS = ['gloss', 'examples', 'hyponyms']              
LINE_SENSES = ['division', 'product', 'cord', 'text', 'formation', 'phone']
HARD_SENSES = ['HARD1', 'HARD2', 'HARD3']
SERVE_SENSES = ['SERVE2', 'SERVE10', 'SERVE6', 'SERVE12']
INTEREST_SENSES = ['interest_1', 'interest_2', 'interest_3', 'interest_4', 'interest_5', 'interest_6']

# relation pairs relative to the provided pos
def define_relpairs(pos=None):
    if pos == None:
        RELS = RELS_ALL
    else:
        if pos == 'n':
            RELS = RELS_NOUNS
        elif pos == 'a':
            RELS = RELS_ADJS
        elif pos == 'v':
            RELS = RELS_VERBS
        else:
            raise ValueError     
    relpairs = [(r1, r2) for r1 in RELS for r2 in RELS]
    return relpairs

# remove stopwords 
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

# remove punctuation
def remove_punctuation(tokens):
    return [word for word in tokens if word not in punctuation]

# obtain the instances where the word has a given sense
def sense_instances(instances, sense):
    return [inst for inst in instances if inst.senses[0] == sense]

# get context
def get_context(inst):
    return " ".join([pair[0] for pair in inst.context])

# converting a nltk.pos_tag tag into a nltk lemmatizer compatible tag
def wordnet_pos(tag):
    if tag.startswith('N'):   
        return 'n'
    elif tag.startswith('V'): 
        return 'v'
    elif tag.startswith('R'): 
        return 'r'
    elif tag.startswith('J'): 
        return 'a'
    else:                     
        return 's'

# computing the gloss of a Synset via the provided relation
def calc_gloss(relation, synset):
    gloss = ''
    if relation == 'gloss':
        gloss = synset.definition()
    elif relation == 'examples':
        gloss = ' '.join(synset.examples())
    elif relation == 'hyponyms':
        for hypo in synset.hyponyms():
            gloss += hypo.definition() + ' '
    elif relation == 'hypernyms':
        for hyper in synset.hypernyms():
            gloss += hyper.definition() + ' '
    elif relation == 'meronyms':
        meronyms = synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms()
        for mero in meronyms:
            gloss += mero.definition() + ' '
    elif relation == 'holonyms':
        holonyms = synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms()
        for holo in holonyms:
            gloss += holo.definition() + ' '
    elif relation == 'also-see':
        for see in synset.also_sees():
            gloss += see.definition() + ' '
    elif relation == 'attribute':
        for att in synset.attributes():
            gloss += att.definition() + ' '
    return gloss

# longest sequence of common words among 2 glosses
def overlap(gloss1, gloss2):
    tmpgloss1 = w_tok(gloss1)
    tmpgloss2 = w_tok(gloss2)
    result = []
    sz = 0
    for i in range(len(tmpgloss1)):
        save = []
        szTemp = 0
        if tmpgloss1[i] in tmpgloss2:
            # the word is also present in gloss2 
            indices = [idx for idx, word in enumerate(tmpgloss2) if word == tmpgloss1[i]]
            # for every occurence
            for idx in indices: 
                temp = i
                save = [tmpgloss1[i]]
                szTemp = 1
                state = True
                # verify how many consecutive words match
                while state:  
                    if temp + 1 < len(tmpgloss1) and idx + 1 < len(tmpgloss2):
                        temp += 1
                        idx += 1
                        nextWord1 = tmpgloss1[temp]
                        nextWord2 = tmpgloss2[idx] 
                        if nextWord1 == nextWord2:
                            szTemp += 1
                            save.append(nextWord1)
                        else:
                            state = False
                            if szTemp > sz:
                                result = save
                                sz = szTemp
                    else:  
                        state = False
                        if szTemp > sz:
                            result = save
                            sz = szTemp
    return result

# calculate score of pair of glosses
def score(gloss1, gloss2):
    score = 0
    topoverlap = ['.']
    gloss1 = ' '.join(remove_punctuation(w_tok(gloss1))).lower()
    gloss2 = ' '.join(remove_punctuation(w_tok(gloss2))).lower()
    while topoverlap != []:
        topoverlap = overlap(gloss1, gloss2)
        if remove_stopwords(topoverlap) != []:
            score += len(topoverlap)**2
        topovlpstr = ' '.join(topoverlap)
        gloss1 = gloss1.replace(topovlpstr, '*')
        gloss2 = gloss2.replace(topovlpstr, '/')
    return score

# sim_score score between 2 synsets
def sim_score(synset1, synset2, pos=None):
    total = 0
    for (r1, r2) in define_relpairs(pos):
        synset1gloss = calc_gloss(r1, synset1)
        synset2gloss = calc_gloss(r2, synset2)
        total += score(synset1gloss, synset2gloss)
    return total

# Word sense disambiguation using the Extended Lesk Algorithm
def extended_lesk(word, sentence, window=3, pos=None):
    # perform pre-processing
    sentence = remove_stopwords(remove_punctuation(w_tok(sentence)))
    # perform lemmatizating
    lemmatizer = WordNetLemmatizer()
    tagger = pos_tag(sentence)
    sentence = [lemmatizer.lemmatize(tup[0], wordnet_pos(tup[1])) for tup in tagger]
    # lemmatization on goal word
    if pos == None:
        tgword = pos_tag([word])
        word = lemmatizer.lemmatize(tgword[0][0], wordnet_pos(tgword[0][1]))
        pos = wordnet_pos(tgword[0][1])
    else:
        word = lemmatizer.lemmatize(word, pos)
    # extract the context window from the sentence
    if word in sentence:
        wrd_idx = sentence.index(word)
        if wrd_idx - window < 0:
            window_words = sentence[0 : wrd_idx + window + 1]
        else:
            window_words = sentence[wrd_idx - window : wrd_idx + window + 1]
        # get the synsets of the target word
        senses = wn.synsets(word)
        topsense = senses[0]
        topscore = 0
        for sense in senses:
            if sense.pos() == pos:
                score = 0
                for w in window_words:
                    if w != word:
                        wordsenses = wn.synsets(w)
                        for tmpsense in wordsenses:
                            score += sim_score(sense, tmpsense, pos)
                if score > topscore:
                    topscore = score
                    topsense = sense
    else: 
        return wn.synsets(word)[0]
    
    return topsense

def main():
    sense = extended_lesk('bank', 'The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities.')
    print(sense)
    print(sense.definition())
    sense = extended_lesk('cone', 'pine cone')
    print(sense)
    print(sense.definition())

if __name__ == "__main__":
    main()
    