## Instructions for preparing embeddings

Download and extract the pretrained common crawl 300D word vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip.
Use the script `embedding.py` to generate the embedding matrix and word indices. The usage is as follows:

```
$ python embedding.py -address address-of-extracted-glove-file
```
