# Layerwise Relevance Visualization in Convolutional Text Graph Classifiers 
Code accompanying paper
```
@inproceedings{Schwarzenberg_lrv_2019,
  title = {Layerwise Relevance Visualization in Convolutional Text Graph Classifiers},
  booktitle = {Proceedings of the EMNLP 2019 Workshop on Graph-Based Natural Language Processing},
  author = {Schwarzenberg, Robert and Hübner, Marc and Harbecke, David and Alt, Christoph and Hennig, Leonhard},
  location = {Hong Kong, China},
  year = {2019}
}
```
![screenshot](https://github.com/DFKI-NLP/lrv/blob/master/data/explanations/lrv.png)
## Approach

Apply LRP to GCNs. Visualize relevance in each layer.

## Installation 
0. Create environment w/ python 3.6, e.g. 
```
conda create --name research-xgcn python=3.6
source activate research-xgcn
```

1. Install requirements.txt, e.g. 

```
pip install -r requirements.txt 
```

2. Install English scispacy language model

```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_sm-0.2.0.tar.gz
```

3. Download (last accessed 2019-05-16)

```https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip```

and extract

```wiki-news-300d-1M.vec```

into 

```./data/embeddings/``` 

### Data

The input data is contained in 

```./data/PubMed_20k_RCT/```  

It was downloaded on 2019-05-16 from 

https://github.com/Franck-Dernoncourt/pubmed-rct

### Experiments 
1. Check config.json, to run the full pipeline, all values in the pipeline namespace should be true.
2. Run main.py which runs 
    - preprocess.py (prepares dictionary, input data)
    - train.py (train XGCN)
    - explain.py (LRV on the trained XGCN, writes explanations to json lines)
    - postprocess.py (summarizes occlusion experiments in plots, writes latex document)

3. run pdflatex on data/explanations/explanations.tex (make sure tiks-dependency is installed)
4. View the explanations in ./data/explanations/explanations.pdf (if config was not changed)




