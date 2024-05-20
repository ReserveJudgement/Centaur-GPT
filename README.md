# Centaur-GPT
## Introduction
This repository contains code, models and data for the paper ....


## Game Generation
Training data is generated from games between chess engines using python chess package. 
The networks used for Maia and Leela can be found in the models folder.
Engines for running them can be downloaded from https://lczero.org/
Stockfish engines (in the paper we used version 11) can be downloaded from https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI
Start states for the games can be found in the opening-positions folder.
The code for generating games can be found in the code folder: Generate-Games.py.
Positions are recorded as FEN strings, and game results from those positions are scored as 0 for loss, 0.5 for draw and 1 for win. 
While this is sufficient for the training objective, other datapoints are stored in csv file (ply count, recommended moves etc.). 


## Training
Given a file of training data, a transformer model can be trained using the Chess-GPT-trainer.py file in the code folder.

![image](https://github.com/ReserveJudgement/Chess-GPT/assets/150562945/101224f5-a510-453f-857a-e4b7068b14d4)

The model receives a fixed-size set of tokens ("context window") as input, and tokens are drawn from a closed vocabulary. So we need to define the context window and vocabulary in order to tokenize.  
Vocabulary consists of 13 possible tokens for each possible status of a square (it can host one of 12 pieces or be an empty square). Positional encoding is added to designate location of each square on the board. Extra tokens appended at the end of the sequence to signify color being played, castling rights of each side and whether the board is in check (even though this could be inferred by the model, it doesn't hurt to add).  
Last token is an auxiliary "CLS" token used to aggregate information from the other tokens, and a classfier is placed on top of it with a sigmoid at the end.
Training is vanilla binary classification with cross-entropy loss.
Model architecture: number of layers: 10, self-attention heads: 16, and embedding size for each token: 128. Final classification is with 3-layer FC network.
Andrey Karpathy's miniGPT implementation is used as a base: https://github.com/karpathy/minGPT  

## Model with Hand-Crafted Features
To train a FC network that takes hand-crafted features, the features first need to be extracted.
This is done with the code in the features.py file.
After that the train-fc.py file trains a model.

## Evaluation
Given a trained model and the base chess engines, the team can be evaluated using the Evaluate.py file.
