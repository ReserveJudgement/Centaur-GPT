# Chess-GPT
## Introduction
This is an experimental encoder for chess using a GPT style transformer model.
Feature encodings for chess deep learning models come in two main categories:  
- Binary Vector for FC Network: the input has the form of a single long vector designed for feeding into a fully connected network. Eg. "bitboard" representation encodes each square on the board with a one-hot vector of 12 bits, each bit representing a possible piece occupying the square, and all zeroes represents an empty square. 64 such vectors are concatenated to make a 768 length binary vector. Extra information can be added at the end, eg. a bit to signify whether there are castling rights. Stockfish, which has a shallow 4-layer FC network for it's evaluation function, uses a more complex overparameterized representation with over 8k bits. But it is still a binary vector representation for a FC network.
- Grid Planes Encoding for CNN: this is used in implementations such as Alpha-Zero. A set of 8x8 grids are one-hot encoded with the presence of certain pieces, eg. a grid with ones for positions of white pawns and zeroes elsewhere. Each grid is a plane, or equivalent of a 'channel' in image data. A stack of such planes represents all the information on the board. Due to the grid/channel form it makes a suitable input to a convolutional neural network, as used in Alpha-Zero.  

Given that transformer networks have since proven effective models in both NLP and visual tasks, this begs the question whether transformers can make a good encoder.
Some have naturally thought of this in the context of chess games as sequences, using chess moves as tokens. But in principle, chess is Markovian, so the sequential history of the game should not matter. Instead, it is proposed to tokenize the squares on the board. 
## Implementation
As a starting point, Andrey Karpathy's miniGPT implementation is used: https://github.com/karpathy/minGPT  
Tokenization is accomplished with positional encoding to designate each square definitively (so context window is 64), and a dictionary of 13 possible tokens for each possible status of the square (hosting one of 12 pieces or an empty square). It is of course possible to add extra information at the end of the sequence with additional tokens to signify things like castling rights (not implemented here).  
Data is taken from simulated games between other strong engines using python chess package. Positions are FEN strings, and game results are 0 for loss, 0.5 for draw and 1 for win, stored in csv file.  
Instead of autoregressive prediction, the output tokens are pooled and a classfier is placed on top with a sigmoid at the end.  
The model objective is to predict chances of winning from a given position. While early positions might be harder to predict, since the game is still open, later positions should be easier since they are less balanced and close to the end.  

## Initial Results
Accuracy (only win/lose games in test set) is 0.78  
TODO: evaluate against engine with 1-depth lookahead (ie. no search algorithm) so that only evaluation functions are being compared  
