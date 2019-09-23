#!/usr/bin/env bash
rsync \
    -rv \
    --copy-links \
    --exclude 'data/archive' \
    --exclude 'data/explanations' \
    --exclude 'data/model' \
    --exclude 'data/embeddings' \
    --exclude 'data/PubMed_20k_RCT' \
    --exclude 'data/logs' \
    --exclude '**__pycache__**' \
    --exclude '.idea' \
    ~/repos/active/research-xgcn/ \
    schwarzenberg@serv-9203.kl.dfki.de:~/repos/active/research-xgcn