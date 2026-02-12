# Question Answering System using DistilBERT

This repository contains a Jupyter notebook that demonstrates how to build and evaluate a Question Answering (QA) system using DistilBERT, a smaller and faster version of BERT. The system is fine-tuned on the Stanford Question Answering Dataset (SQuAD).

## Abstract

This project provides a practical implementation of a Question Answering system using a pre-trained DistilBERT model. The goal is to fine-tune the model on the SQuAD dataset to answer questions based on a given context. This notebook walks through the entire process, from data loading and preprocessing to model training, and evaluation.

## Features

*   **State-of-the-Art Model:** Utilizes the DistilBERT model, a smaller, faster, and lighter version of BERT, which is ideal for production environments.
*   **Standard Dataset:** Uses the well-known SQuAD dataset for training and evaluation.
*   **End-to-End Pipeline:** Provides a complete pipeline for a QA system, including:
    *   Data loading and preprocessing.
    *   Tokenization and feature extraction.
    *   Model fine-tuning.
    *   Prediction and evaluation.

## How to Use

### Prerequisites

To run the notebook, you need to have Python 3 and the following libraries installed:

*   transformers
*   torch
*   matplotlib
*   seaborn
*   numpy
*   pandas
*   datasets
*   evaluate

You can install the required libraries using pip:

```bash
pip install transformers torch matplotlib seaborn numpy pandas datasets evaluate
```

### Data

The notebook uses the SQuAD dataset, which is automatically downloaded and loaded using the `datasets` library.

### Running the Notebook

1.  Clone this repository to your local machine.
2.  Open the `Question_answering_system.ipynb` notebook in a Jupyter environment.
3.  Run the cells in the notebook sequentially to see the entire process of building and evaluating the QA system.

## Model

### Architecture

The QA system is based on the `distilbert-base-uncased` model. This model is a smaller version of BERT that is faster and uses less memory, while still providing comparable performance on many NLP tasks.

### Training

The model is fine-tuned on the SQuAD dataset. The notebook demonstrates how to preprocess the data, set up the training arguments, and train the model.

### Evaluation

The performance of the fine-tuned model is evaluated on the SQuAD validation set using the following metrics:

*   **Exact Match (EM):** Measures the percentage of predictions that match the ground truth answers exactly.
*   **F1 Score:** Measures the harmonic mean of precision and recall, providing a more robust evaluation than EM.
