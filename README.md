# NLP Final Project

This repository contains our team's final project for the Natural Language Processing for Data Science course.

## Project Title:
Exploring Neural Approaches for Tiny Story Generation 


## Team Members

- Abhilasha Singh
- Pranjal Wakpaijan
- Shabnam Rafat Ummey Sawda

## Problem Selection and Motivation

Automated story generation is a challenging task requiring models to maintain narrative coherence, vocabulary alignment, and contextual consistency. This project investigates how different neural architectures perform in the task of generating short children’s stories based on a given prompt and a set of required keywords.


## Repository Structure

```
Code/
├── Result/
├── component/
│   ├── __init__.py
│   ├── gnn_encoder.py
│   ├── gnn_retrieval_slm.py
│   ├── retriever.py
│   ├── utils.py
│   ├── utils_gnn.py
│   └── utils_slm.py
├── Claude.py
├── EDA.py
├── Flan-T5.py
├── LSTM.py
├── Readme.md
├── baseline_model.py
├── build_graph_dataset.py
├── evaluate_gnn_retrieval_slm.py
├── evaluate_slm.py
├── evaluation_metric.py
├── tokenize_data.py
├── train_gnn_retrieval_slm.py
├── train_slm.py
└── train_test_val_split.py

```



# Data Set

We will use the TinyStories dataset from Hugging Face, a collection of synthetically generated short stories created specifically for
training and evaluating language models on simple, coherent narratives.


Source:
https://huggingface.co/datasets/roneneldan/TinyStories





