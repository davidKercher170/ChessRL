# ChessRL
Chess model created using Deep Reinforcement Learning. Details the use of novel attention mechanisms and a new paradigm for transfer learning.

### Channel Attention
I use a variation of Efficient Channel Attention (https://arxiv.org/abs/1910.03151). In addition to the Average Pooling layer across the spatial dimension, I add a Max Pooling layer. This has been done in similar approaches (such as CBAM).

### Spatial Attention
As with all decisions made with the model, I wanted an intuitive spatial attention mechanism. This naturally led to examining spatial context relative to possible piece movements such as diagonals, rows, columns, and knight moves. I eventually decided, for simplicity, to focus spatial attention on row and column relationships. In my research, I later found that this is nearly identical to the proposed strip pool mechanism (https://www.sciencedirect.com/science/article/abs/pii/S0098300421002259).

### Value Head
The LeelaZero and AlphaZero model value heads consist of:

convolutional reduction layer down to a 1-16 filters \longrightarrow fully connected layer of 128 nodes -> fully connected layer with 1 node -> tanh activation (scale between -1 and 1) 

I opted for a different approach (that quite honestly felt more intuitive and more interesting). My approach is: 

8x8 Kernel Depthwise Convolutional layer (reduce each channel down to a single value) -> Fully Connected Layer of 128 Nodes -> Fully Connected Layer with 1 Node. 

The idea is to produce a single value for each pattern (channel) that represents the current opportunities and hazards of that specific point of view.

### Positional Attention
I used a weighted 8x8 matrix as an additional channel concatenated to each input. In theory, this should represent positional attention. Positional attentions gives focuses the model on important squares of the board irrelevant of the current state and spatial context. For example, the middle squares are far more important than edge squares. Over the course of training, the model learns a representation of the most vital squares to control, attack, and defend.

### Partial Convolutional Based Padding
The board representation for a chess state is relatively small (in height and width) vs. the typical use of residual convolutional blocks in object detection and image analysis. As such, every square of the board holds vital context on the current state of the game. I opted for partial convolutional padding over zero padding as it maximizes boundary context. More information on Partial Convolutional Padding can be found in the publication (https://arxiv.org/abs/1811.11718).
