#!/bin/bash
python3  main.py --epochs 4 --emb 20news.csv
python3  main.py --epochs 4 --emb urban_dictionary.csv
python3  main.py --epochs 4 --emb maas_imdb.csv