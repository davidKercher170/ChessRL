# ChessRL
Chess model created using Deep Reinforcement Learning. Details the use of novel attention mechanisms and a new paradigm for transfer learning. I follow the protypical structure layed out in the LeelaZero and AlphaZero (https://arxiv.org/abs/1712.01815) models with a stack of residual convolutional blocks, each with a 3x3 kernel and 128 filters.

### Board Representation
I use an 8x8x18 representation of the current board state as input to the model. The height and width represent the chess board spatial dimensions, while the 18 channels represent characteristics of the board. Channels 1-6 contain information for each piece type (Pawn, Knight, Bishop, etc...) for white and Channels 7-12 hold information for each piece type for black. Channels 13-17 contain information for king side and queen side castling rights for each player. Channel 18 is set to 1 if the board has been flipped and otherwise is set to 0 (for training, the spatial information for all states are represented from the white perspective). As defined this is a typical approach to chess reinforcement learning. More advanced models will contain temporal conext with previous states channel.

### Channel Attention
I use a variation of Efficient Channel Attention (https://arxiv.org/abs/1910.03151). In addition to the Average Pooling layer across the spatial dimension, I add a Max Pooling layer for improved information flow. This has been done in similar approaches (such as CBAM). After producing a value for each channel, I intermix channel information, seperately for the two pooling layers, with a 3x3 Kernel Convolution, add the results together, and then activate with a sigmoid function. Lastly, we scale the resulting channel values by a learnable gating mechanism to determine how significant the attention contribution should be and then multiply the input by the resulting values.

### Spatial Attention
As with all decisions made with the model, I wanted an intuitive spatial attention mechanism. This naturally led to examining spatial context relative to possible piece movements such as diagonals, rows, columns, and knight moves. I eventually decided, for simplicity, to focus spatial attention on row and column relationships. In my research, I later found that this is nearly identical to the proposed strip pool mechanism (https://www.sciencedirect.com/science/article/abs/pii/S0098300421002259). The idea is to use $1x8$ Kernel and $8x1$ Kernel convolutions to produce a value for all positions in the same columns and rows respectively, examining all filters to produce a single value for each row, column. Then we combine these values together into a matrix of form $A_{ij} = row_i + column_j$. We fuse these values together using a $3x3$ kernel depthwise convolution with a sigmoid activation and then scale our original input by the resulting values. Once again, a gating mechanism is used to scale the contribution the spatial attention.

### Value Head
The LeelaZero and AlphaZero model value heads consist of:

$3x3$ Kernel Convolutional reduces to 1-16 filters $\longrightarrow$ Fully Connected layer, 128 nodes $\longrightarrow$ Fully Connected layer, 1 node $\longrightarrow$ Tanh Activation (scale between -1 and 1).

I opted for a different approach. The idea is to produce a single value for each channel (pattern) that represents a win/loss contribution. My approach is: 

$8x8$ Kernel Depthwise Convolutional (reduce each channel down to a single value) $\longrightarrow$ Fully Connected Layer of 128 Nodes $\longrightarrow$ Fully Connected Layer, 1 Node, Tanh Activation. 

### Positional Attention
I used a weighted 8x8 matrix as an additional channel concatenated to each input. In theory, this should represent positional attention. Positional attentions gives focuses the model on important squares of the board irrelevant of the current state and spatial context. For example, the middle squares are far more important than edge squares. Over the course of training, the model learns a representation of the most vital squares to control, attack, and defend.

### Partial Convolutional Based Padding
The board representation for a chess state is relatively small (in height and width) vs. the typical use of residual convolutional blocks in object detection and image analysis. As such, every square of the board holds vital context on the current state of the game. I opted for partial convolutional padding over zero padding as it maximizes boundary context. While typical zero padding will compute non-zero values for boundary positions, it will normalize by the full size of the kernel. Partial convolutional padding will only normalize by the number of non-zero weights considered. In effect, zero-padding diminishes the value of boundary values in comparison to non-boundary values. More information on Partial Convolutional Padding can be found in the publication (https://arxiv.org/abs/1811.11718).

# Transfer Learning
Transfer learning is the process of applying the knowledge in a pretrained model to a new dataset/problem. In the case of chess, I will train a base model through IQL on a large database and then flesh out learning done through self-play with a Monte Carlo Tree Search. Once the model is performing sufficiently well, I then use this base model to create submodels thats mimic different styles.
