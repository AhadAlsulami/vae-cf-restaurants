# CF Implementation for Restaurant Recommendation Systems | Fork of "Variational Autoencoders for Collaborative Filtering"
The provided repository contains the implementation of a restaurant recommendation system based on collaborative filtering. It utilizes Variational Autoencoders (VAEs) and was originally developed as part of a Recommender Systems course project. The code in the repository is a fork from the code associated with the research paper titled "[Variational autoencoders for collaborative filtering]([url](https://arxiv.org/pdf/1802.05814.pdf))" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, presented at The Web Conference (WWW) 2018. The repository has been subsequently updated by Jakub Kuniszewski.

## Project Members
* Ahad Alsulami | 2008263
* Shatha Alotaibi | 2006590
* Reema Albishri | 2007876

## Implementation
The code is written in Python and utilizes various libraries and frameworks, including TensorFlow. The main steps of the implementation include:
1. Importing necessary libraries
2. Preprocessing the data
3. Building the Variational Autoencoder model
4. Training the model
5. Rvaluating the generated restaurant recommendations

## Repository Structure
The repository consists of the following files:
* RS_Project.ipynb: This Jupyter Notebook contains the code implementation for the project.
* Restaurant reviews.csv: This dataset file contains restaurant reviews. The dataset was sourced from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/batjoker/zomato-restaurants-hyderabad/data).

_______________________


# Original README
This repository is a fork of the code originally accompanying the paper 
"[Variational autoencoders for collaborative filtering](https://arxiv.org/abs/1802.05814)" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018."

See last section for the original README contents

## Introduction

This fork of the repository aims to update the original code, which was designed for older versions of Python and TensorFlow, making it compatible with the latest Python and TensorFlow versions, particularly for use in platforms like Google Colab.

## Changes Made

- Refactored Pandas data preparation code cells to be executable
- Preinstall `bottleneck` package used in this notebook
- Add cell for fetching and unzipping the dataset
- Fix hyperparameter setup
- Fix MultiDAE and MultiVAE regularization step
- Add `tqdm` for more readable training tracking
- Add 1e-6 to divisors when calculating NDCG and Recall to fix NaN values issue
- Attached `requirements.txt`

## Evaluation

The notebook should be ready to go as is in Google Colab (T4 GPU instance recommended, available in free version of Google Colab).
Executing all code in notebook takes about 4 hours in T4 instance.

The results of validation NDCG per epoch, test NDCG and test Recall are not exactly the same like those displayed in the original notebook, but are very close.

## Main package versions (from requirements.txt)

- Pandas - 1.5.3
- Tensorflow - 2.14.1

# Original README
# Variational autoencoders for collaborative filtering

This notebook accompanies the paper "[Variational autoencoders for collaborative filtering](https://arxiv.org/abs/1802.05814)" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.

In this notebook, we show a complete self-contained example of training a variational autoencoder (as well as a denoising autoencoder) with multinomial likelihood (described in the paper) on the public Movielens-20M dataset, including both data preprocessing and model training.
