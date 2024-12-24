# Centaur-GPT
![image](https://github.com/user-attachments/assets/f7db2678-414d-463a-9760-24303411c4ad)

## Introduction
This repository contains code, models and data for the paper "Modeling the Centaur: Human-Machine Synergy in Sequential Decision Making".
It is used to model a "team" of chess engines, consising of at least two base players, and a "manager" that decides at each position which player makes the move on behalf of the team. 
This setup is used to model a "centaur" team, in which the base players are Maia-Chess, a humanlike network, and Leela-Chess-Zero, a pure self-play RL-trained network.

## Chess Engines
Maia networks can be downloaded from [https://github.com/CSSLab/maia-chess].
The Maia network used in the paper is 1900 ELO and can be found in the team-players folder. 

Leela networks can be found at [https://training.lczero.org/networks/].
The one used for the symmetric team can be found in the team-players folder.

The networks used for Maia and Leela should be placed in seperate folders, each with the LC0 engine.
The LC0 engine can be downloaded from [https://lczero.org/play/download/)]

Stockfish engines can be downloaded from https://drive.google.com/drive/folders/1nzrHOyZMFm4LATjF5ToRttCU0rHXGkXI


## Generating Games
Training data is generated from games between chess engines using the python chess package [https://python-chess.readthedocs.io/en/latest/]. 
Start states for the games can be found in the "opening-positions" folder.
The code for generating games can be found in the "code" folder: Generate-Games.py.
Running games requires specifying team members and adversary, as stored in a global dictionary with paths to the respective engines.
It also requires creating a "manager" object (see below) which scores the recommendations and decides on a move for the team at each disagreement.
Output is a csv file. Positions are recorded as FEN strings, and evaluations of the team players used for each position are scored as 0 for estimated loss if that player is chosen, 0.5 for estimated draw and 1 for estimated win. While this is sufficient for the training objective, additional data is also stored in the file (ply count, recommended moves etc.). 


## Manager types
Classes of manager for the virtual centaur team:
- The "CentaurModel" class loads a torch model, either transformer or fully-connected, and uses it to make decisions at each position.
- The "PolicyIterate" and "FCIterate" classes also load a torch model (transformer and fully-connected respectively), but they score positions by playing out each recommendation of the base players and then rolling out the rest of the game, using the model. These classes are good for reinforcement learning using the policy iteration algorithm (see illustration at head of file).
- The "Oracle" class scores the recommendations without using a model, by rolling out games from recommendations, using a specified base player alone for continuation.
  This is for comparison to an upper baseline, since this type of rollout at each move is too computationally intensive for actual play.
- The "RandomChoice" class runs a random mixture policy between the base players. By default p=0.5, but it can be adjusted. Setting p=0 or p=1 can be used to run the baselines of each team member playing alone.
- The "Expert" class loads an additional chess engine, which scores the recommendations of the team members.


## Training
Given a file of training data, a transformer model can be trained using the CentaurGPT-trainer.py file in the "code" folder.
The board position is encoded using 64 tokens to represent the status of each square on the board, and an additional set of tokens to represent castling rights, whether there is check and a classification token:

![image](https://github.com/user-attachments/assets/56cf1751-ab03-4940-bd93-7ec971416282)

Training data is converted into a torch dataset using RelAdvantage class.
The trainer stores the trained transformer encoder and the final classifier layer as separate models, one with the "Encoder" suffix and the other with the "Clf" suffix.
When generating games using the trained model, both need to be used.
Trained models for the symmetric team can be found in the "models" folder.


## Hand-Crafted Features
To extract handcrafted features of a chess board position, the Features.py file can be used.
Extracted features are saved in a csv file, with the positions as FEN strings, a column for each feature and the evaluations preserved in the last column.
This can be used to train a fully connected network using a small number of features.


## Evaluation
To evaluate a trained manager model, generate games using the test-opening-positions.csv as start states, and using the CentaurModel class for the manager object.
Records of games generated to evaluate models from the "models" folder, can be found in the "evaluations" folder.


## Explainability
Explainability analysis can be done using the helper functions in the Explainability.py file.
Functions include:
- getting attention scores from model over a dataset
- querying the attention scores for the variables of interest (pieces vs empty squares, attacked pieces vs not-attacked, move comparisons)
- heatmap visualization of attention scores in the transformer model over board positions
- calculate non-parametric Aw effect size

Examples of heatmap of attentions scores from the transformer model manager:
![image](https://github.com/user-attachments/assets/d871dce9-e5bb-4a40-8149-4cc1284519b5)


## Results

![image](https://github.com/user-attachments/assets/545f41b5-7c1e-4d03-a4b9-ead86f3daef9)

The RL-trained manager with the transformer architecture, manages to produce synergy, and also outperfrom the "expert" manager. 
The oracle has significantly higher performance than other manager types, indicating that there is still ample headroom for additional synergy.


