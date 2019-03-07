# Copyright 2019 Saarland University, Spoken Language Systems LSV 
# Author: Michael A. Hedderich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS*, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
#
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

"""
This script creates the reverse dictionary dataset used in 

Michael Aloys Hedderich, Andrew Yates, Dietrich Klakow and Gerard de Melo: 
Using Multi-Sense Vector Embeddings for Reverse Dictionaries.
13th International Conference on Computational Semantics (IWCS 2019)

When using this script or dataset, please make sure to cite us.

The script creates csv files containing the training, development
and test set. The format ist <target word>;<tokenized description>\n

It also creates corresponding lists of Instance objects
for using the datasets directly in your code without having to parse
the csv files.

Before running the script, please make sure to install it properly by
using Python 3.6.5 and installing the package dependency nltk in version 3.3
and spacy in version 2.0.11. You also need to download the English language pack
en_core_web_sm-2.0.0 for Spacy.

pip install nltk==3.3
pip install spacy==2.0.11
python -m spacy download en_core_web_sm
"""

import random
import nltk
nltk.download("wordnet") # Downloading the wordnet corpus
from nltk.corpus import wordnet as wn

import spacy
from spacy.lang.en import English

# Making sure that the versions are exactly the same
assert nltk.__version__ == "3.3"
assert wn.get_version() == "3.0"
assert spacy.__version__ == "2.0.11"

def main():
    # loading the tokenizer
    spacy_nlp = spacy.load("en_core_web_sm")
    tokenizer = English().Defaults.create_tokenizer(spacy_nlp)
    
    # Obtaining all the synsets and splitting them into train, dev and test. 
    # Splitting along synsets (and not instances) is important to not taint the test data.
    all_synsets = list(wn.all_synsets())
    random.seed(742382)
    random.shuffle(all_synsets)

    # 0.8/0.1/0.1 train/dev/test split
    split_index_train_dev = int(len(all_synsets) * 0.8)
    split_index_dev_test = int(len(all_synsets) * 0.9)

    train_synsets = all_synsets[:split_index_train_dev]
    dev_synsets = all_synsets[split_index_train_dev:split_index_dev_test]
    test_synsets = all_synsets[split_index_dev_test:]
    
    # Filter lists are used to exclude target words that are not in the embeddings
    
    # Filtering 19 lemmas that are not part of DeConf
    # i.e. synset.lemmas() -> lemma.key() not in https://pilehvar.github.io/deconf/ -> sense_key_map.txt
    filterlist_deconf = load_filterlist("filterlist_deconf.txt")

    # Filtering 32718 words that are not part of the pretrained word2vec embedding
    # from https://code.google.com/archive/p/word2vec
    filterlist_word2vec = load_filterlist("filterlist_word2vec.txt")
    
    # Converting the synsets into instances
    train_instances = convert_synsets_into_instances(train_synsets, tokenizer, filterlist_deconf, filterlist_word2vec)
    dev_instances = convert_synsets_into_instances(dev_synsets, tokenizer, filterlist_deconf, filterlist_word2vec)
    test_instances = convert_synsets_into_instances(test_synsets, tokenizer, filterlist_deconf, filterlist_word2vec)

    # Just some tests to make sure everything went well
    assert len(train_instances) == 85136
    assert len(dev_instances) == 10521
    assert len(test_instances) == 10502

    assert train_instances[0].word == "heroism" 
    assert dev_instances[-1].word == "dissolve"
    assert test_instances[0].word == "dependent"
    
    # You can now use the Instance objects in train_instances, dev_instances
    # and test_instances directly in your code. Use instance.word to 
    # get the target word and instance.description for the tokenized
    # description.
    
    # Alternatively, the datasets are written to files in the csv format.
    write_instances_to_file(train_instances, "train.csv")
    write_instances_to_file(dev_instances, "dev.csv")
    write_instances_to_file(test_instances, "test.csv")

class Instance:
    """ An instance consisting of a (target) word and a list of description words
    """
    def __init__(self, word, description):
        """ Target word and tokenized description
        """
        self.word = word
        self.description = description
        
    def to_csv(self):
        escaped_description = " ".join(self.description).replace(";","\\;")
        return "{};{}".format(self.word, escaped_description)
   
    def __str__(self):
        return "{}: {}".format(self.word, self.description)
   
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.word == other.word and self.description == other.description
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple([self.word, self.description]))
    
def tokenize_sentence(tokenizer, sentence):
    """ Tokenizes and lowercases the description using the English 
        Spacy tokenizer
    """
    tokens = tokenizer(sentence)
    words = [token.text for token in tokens]
    words = [word.lower() for word in words]
    return words
    
def load_filterlist(path):
    """ Filter lists are used to exclude target words that are not in
        the embeddings
    """
    with open(path) as input_file:
        filterlist = input_file.read().splitlines()
    return set(filterlist)
    
def convert_synsets_into_instances(synsets, tokenizer, 
                                   filterlist_deconf, filterlist_word2vec):
    """ A synset can have multiple lemmas. For each lemma an Instance(lemma, description)
        is created.
    """
    random.seed(846271)
    instances = []
    for synset in synsets:
        description = synset.definition()
        description = tokenize_sentence(tokenizer, description)
        
         # filter all multi-word phrases indicated by _
        lemmas = [lemma for lemma in synset.lemmas() if "_" not in lemma.name()]
        if len(lemmas) == 0:
            continue
        
        for lemma in lemmas:
            # filter lemmas not supported by DeConf
            if lemma.key() in filterlist_deconf:
                continue
                
            # filter lemmas not supported by the pretrained word2vec embedding
            if lemma.name() in filterlist_word2vec:
                continue
            
            new_instance = Instance(lemma.name(), description)
            new_instance.wn_synset = synset
            new_instance.wn_lemma_key = lemma.key()
            instances.append(new_instance)
            
    return instances
    
def write_instances_to_file(instances, path):
    """ Write the given list of instances to a file (in csv format)
    """
    with open(path, "w") as output_file:
        for instance in instances:
            output_file.write(f"{instance.to_csv()}\n")
            
if __name__ == "__main__":
    main()
