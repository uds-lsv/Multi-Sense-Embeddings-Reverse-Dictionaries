# Using Multi-Sense Vector Embeddings for Reverse Dictionaries

This repository contains additional material for the publication

*Michael Aloys Hedderich, Andrew Yates, Dietrich Klakow and Gerard de Melo:*  
**Using Multi-Sense Vector Embeddings for Reverse Dictionaries.**  
To appear in the Proceedings of the 13th International Conference on Computational Semantics (IWCS 2019)

## Reverse Dictionary Dataset
A reverse dictionary is a tool for authors and writers seeking a word that is *on the tip of their tongue*. Among other uses, it is also an interesting task for sequence embedding models. This dataset maps a definition or descriptions of a word to the word it describes. More details can be found in the paper (sections 2.2 and 4.1).

We provide a script to obtain this dataset. The script creates csv files containing the training, development and test set. The format ist *<target word>;<tokenized description>\n*. It also creates corresponding lists of Instance objects for using the datasets directly in your code without having to parse the csv files.

Before running the script, please make sure to install it properly by using Python 3.6.5 and installing the package dependency nltk in version 3.3 and spacy in version 2.0.11. You also need to download the English language pack en_core_web_sm-2.0.0 for Spacy. Then run the Python script. This can be done with the following commands:

```
pip install nltk==3.3
pip install spacy==2.0.11
python -m spacy download en_core_web_sm
python dataset_creation.py
```

You should obtain three files: train.csv, dev.csv and test.csv. For directly using the Instance objects, take a look at the code in dataset_creation.py

## Figures

All the plots, including the visualizations of the ambiguous word vectors, can be found in the subdirectory *figures* as PDF files.

## Study of Senses and Attention

We performed a small study to gain more insight into the different senses occurring in the input sequences as well as into the learned attention. We manually labeled 275 words that are part of the input description. Out of these, 100 (37%) had one sense of the multiple possible meanings provided by the multi-sense embedding. More details can be found in section 4.7 of the paper.

The raw results of this study can be found in the file *evaluation_senses_attention.json*. Each line is one instance (consisting of a target word and its description). For each token in the description, the embedding type is given (multi-sense, single-sense fallback or oov). For tokens where a multi-sense embedding can be used, the manually labeled correct sense according to Wordnet is given as well as the attention the model gave each sense.

The fields in the json file have the following meaning

```
"target_word": the target word for this instance
"description": list of tokens in the description
-- "token": the token
-- "token_idx": index of the token in the description
-- "embedding_type": embedding_type used (multi-sense, single-sense fallback or oov)
-- "attention": attentions assigned by the model to each multi-sense vector

if embedding_type == "multi-sense-embedding":
-- "num-senses" = number of available senses according to deconf senses
-- "sense-indices" = selected indices of the deconf senses for a given word (only important if the number of senses > max number of senses)
-- "sense-deconf_keys": keys of the sense vectors according to DeConf
-- "selected-sense-wordnet-synset-and-lemma-keys": corresponding wordnet and lemma keys of the sense vectors
                
if num-senses > 1:
-- "correct-sense-index": manually labeled index of the correct DeConf sense (-1 if no sense applies)
-- "correct-sense-wordnet-synset": manually labeled correct Synset in WordNet for this token in this context
```

## Citation

If you use any of these resources in your work, please cite us.

```
@inproceedings{MultiSenseReverseDictionary:2019,
  author = {Michael Aloys Hedderich and Andrew Yates and Dietrich Klakow and Gerard de Melo},
  title = {Using Multi-Sense Vector Embeddings for Reverse Dictionaries},
  year = 2016,
  booktitle = {To appear in the Proceedings of the 13th International Conference on Computational Semantics (IWCS 2019)},
}
```
