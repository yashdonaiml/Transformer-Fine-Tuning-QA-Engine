# Question Answering System using BERT
This repository contains the code for developing and evaluating a Question Answering (QA) system based on BERT (Bidirectional Encoder Representations from Transformers). The system is trained and evaluated on the Stanford Question Answering Dataset (SQuAD) and utilizes the DistilBERT model architecture.

## Abstract
In the realm of Natural Language Processing (NLP), Question Answering (QA) systems play a pivotal role in facilitating human-computer interaction and information retrieval. This project proposes the development of an advanced QA system harnessing the power of BERT and trained on the SQuAD dataset. Through rigorous experimentation and evaluation, the aim is to explore the strengths and limitations of the BERT-based QA approach and contribute valuable insights to the broader NLP community.

## How to Access Necessary Data
The necessary data for this project, including the SQuAD dataset, can be accessed from the official sources. SQuAD data can be downloaded from the SQuAD website.

Dataset Link: https://huggingface.co/datasets/rajpurkar/squad 

## How to Install and Configure the Code
- Clone this repository to your local machine

- Install the required dependencies

- Configure the code by modifying the necessary parameters in the configuration files provided.

## How to Retrain the Model to Reproduce Experiments

To retrain the model and reproduce the experiments:

- Prepare the data: Follow the steps outlined in the experiment section to load, filter, and preprocess the SQuAD data.
- Train the model: Use the provided scripts to train the BERT-based QA model on the prepared data.
- Evaluate the model: After training, evaluate the model using the provided evaluation scripts to assess its performance on the validation set.
- Analyze results: Conduct error analysis and interpretation of the model's predictions to identify areas for improvement.
  
## How to Run the Trained Model on New Data

To run the trained model on new data:

- Load the model: Load the saved model weights and configuration.
- Preprocess the new data: Tokenize the questions and contexts using the DistilBERT tokenizer, ensuring the input format is compatible with the model.
- Inference: Perform inference using the loaded model on the preprocessed data to generate predictions.
- Evaluate: Evaluate the model's performance on the new data using appropriate metrics.
