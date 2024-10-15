# Centaur-GPT
## Introduction
This repository contains code, models and data for the paper "Modeling the Centaur: Human-Machine Synergy in Sequential Decision Making".
It is used to model a "team" of chess engines, consising of at least two base players, and a "manager" that decides at each position which player makes the move on behalf of the team. 
This setup is used to model a "centaur" team, in which the base players are Maia-Chess, a humanlike network, and Leela-Chess-Zero, a pure self-play RL-trained network.

## Chess Engines
The networks used for Maia and Leela can be found in the models folder. They should be placed in seperate folders, each with the LC0 engine.
The Maia network used is 1900 ELO. Other ranked networks can be downloaded from [https://github.com/CSSLab/maia-chess].


The Leela networks used start approximately equal strength to the Maia network and are progressively stronger. Other Leela networks can be found at [https://training.lczero.org/networks/].


Engines for running these models can be downloaded from [https://lczero.org/play/download/)]


Previous Stockfish engines (the paper used version 11 as adversary against the symmetric team, and version 14 against the asymmetric teams) can be downloaded from https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI


## Generating Games
Training data is generated from games between chess engines using the python chess package [https://python-chess.readthedocs.io/en/latest/]. 
Start states for the games can be found in the "opening-positions" folder.
The code for generating games can be found in the "code" folder: Generate-Games.py.
Running games requires specifying team members and adversary, as stored in a global dictionary with paths to the respective engines.
It also requires creating a "manager" object which scores the recommendations and decides on a move for the team at each disagreement.
Different kinds of games are generated with different classes of manager:
- The "CentaurModel" class loads a torch model, either transformer or fully-connected, and uses it to make decisions at each position. This is good for evaluation of a model.
- The "PolicyIterate" and "FCIterate" classes also load a torch model (transformer and fully-connected respectively), but they score positions by playing out each recommendation of the base players and then rolling out the rest of the game using the model. These classes are good for reinforcement learning using the policy iteration algorithm.
- The "Oracle" class scores the recommendations without using a model, by rolling out games with each base player alone.
- The "RandomChoice" class runs a random mixture policy between the base players. By default p=0.5, but it can be adjusted. Setting p=0 or p=1 can be used to run the baselines of each team member playing alone.
- The "Expert" class loads an additional chess engine, which scores the recommendations of the team members.


Output is a csv file. Positions are recorded as FEN strings, and evaluations of the team players used for each position are scored as 0 for loss, 0.5 for draw and 1 for win. 
While this is sufficient for the training objective, additional data is also stored in the file (ply count, recommended moves etc.). 


## Training
Given a file of training data, a transformer model can be trained using the CentaurGPT-trainer.py file in the "code" folder.
Training data is converted into a torch dataset using RelAdvantage class.
The trainer stores the trained transformer encoder and the classifier models separately, one with the "Encoder" suffix and the other with the "Clf" suffix.
When generating games using the models, both need to be used.
Trained models can be found in the "models" folder, with the name "ManagerTransformer".


## Hand-Crafted Features
To extract handcrafted features of a chess board position, the Features.py file can be used.
Extracted features are saved in a csv file, with the positions as FEN strings, a column for each feature and the evaluations preserved in the last column.


## Evaluation
To evaluate a trained manager model, generate games using the test-opening-positions.csv as start states, and using the CentaurModel class for the manager object.
Records of games generated to evaluate models from the "models" folder, can be found in the "evaluations" folder.


## Explainability
Explainability analysis can be done using the helper functions in the Explainability.py file.
Functions include:
- getting attention scores from model over a dataset
- querying the attention scores for the variables of interest (pieces vs empty squares, attacked pieces vs not-attacked, move comparisons)
- heatmap visualization of attentions over board
- calculate non-parametric Aw effect size
- feature importance attribution using captum, for use with a FC model [https://captum.ai/]
