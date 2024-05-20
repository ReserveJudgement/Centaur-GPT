# Centaur-GPT
## Introduction
This is an experimental encoder for chess using a GPT style transformer model.
Existing feature encodings for SoTA chess deep learning models come in two main categories:  
- Binary Vector for FC Network: the input has the form of a single long vector designed for feeding into a fully connected network. Eg. "bitboard" representation encodes each square on the board with a one-hot vector of 12 bits, each bit representing a possible piece occupying the square, and all zeroes represents an empty square. 64 such vectors are concatenated to make a 768 length binary vector. Extra information can be added at the end, eg. a bit to signify whether there are castling rights. Stockfish, which has a shallow 4-layer FC network for it's evaluation function, uses a more complex overparameterized representation with over 8k bits. But it is still a binary vector representation for a FC network.
- Grid Planes Encoding for CNN: this is used in implementations such as Alpha-Zero. A set of 8x8 grids are one-hot encoded with the presence of certain pieces, eg. a grid with ones for positions of white pawns and zeroes elsewhere. Each grid is a plane, or equivalent of a 'channel' in image data. A stack of such planes represents all the information on the board. Due to the grid/channel form it makes a suitable input to a convolutional neural network, as used in Alpha-Zero.  

Given that transformer networks have since proven effective models in both NLP and visual tasks, this begs the question whether transformers could constitute a third paradigm for chess encoding.
Some have naturally thought of this in the context of chess games as sequences, using chess moves as tokens. But in principle, chess is Markovian, so the sequential history of the game should not matter. Instead, it is proposed to tokenize the squares on the board for any given position. 

## Model Architecture
![image](https://github.com/ReserveJudgement/Chess-GPT/assets/150562945/101224f5-a510-453f-857a-e4b7068b14d4)

The model receives a fixed-size set of tokens ("context window") as input, and tokens are drawn from a closed vocabulary. So we need to define the context window and vocabulary in order to tokenize.  
Vocabulary consists of 13 possible tokens for each possible status of a square (it can host one of 12 pieces or be an empty square). Positional encoding is added to designate location of each square on the board. Extra tokens appended at the end of the sequence to signify color being played, castling rights of each side and whether the board is in check (even though this could be inferred by the model, it doesn't hurt to add).  
Last token is an auxiliary "CLS" token used to aggregate information from the other tokens, and a classfier is placed on top of it with a sigmoid at the end.
Training is vanilla binary classification with cross-entropy loss.
Model architecture: number of layers: 10, self-attention heads: 16, and embedding size for each token: 128. Final classification is with 3-layer FC network.
Andrey Karpathy's miniGPT implementation is used as a base: https://github.com/karpathy/minGPT  

## Game Generation
Training data is generated from games between chess engines using python chess package. Positions are recorded as FEN strings, and game results are scored as 0 for loss, 0.5 for draw and 1 for win. While this is sufficient for the training objective, other datapoints are stored in csv file for possible future use (plies, moves, engine scores etc.).  

