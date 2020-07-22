# snake-rl
A deep Q-learning reinforcement learning model for the game snake. Although the model did not turn out very well as the agent can play decently but not perfectly, it was a good opportunity to learn about the basics of reinforcement learning and Q-learning. The game is rendered using pygame and the reinforcement learning agent is built on tensorflow and keras.

Directory structure:
- snakpe.py: The snake game, responsible for handling game logic and returning the current state of the game.
- snake_agent.py: The deep Q-learning agent.
- snake.h5: A saved model that performs decently.

## Training strategy and trials
- Initially the neural network consisted of 3 layers, each consisting of 32 weights and using a reLU activation function. However, trial and error found that using more layers and gradually decreasing the number of weights yielded better results. Hence, the final agent uses 5 hidden layers using a reLU function and with 256, 128, 128, 64 and 32 weights.
- The agent is penalised for dying (colliding with an obstacle) and rewarded according to how long the snake grows.
- I initially tried to train the agent by passing in the entire game board. However, it was not able to effectively learn from this information as there are too many possible states when passing in the entire board and the agent got nowhere.
- I then trained the agent by providing it information about its head location, tail location and the location of the "apple" or goal. This led to much faster training times and allowed the agent to play effectively. The agent was able to consistently achieve reasonable results. However, as it grew longer, it lacked information about the rest of its body and often collided into itself.
- To overcome this, I provided the agent the entire board alongside the head and tail. Once more, this was too much information and there were too many states for the model to train effectively.
- I then attempted to provide the agent the head, tail, apple location, direction of movement and length. This trained effectively and achieved better results than when just the head, tail and apple are provided and led to moderate results. However, there were still scenarios where it collided into itself and it sometimes got stuck in infinite loops. snake.h5 contains a model trained using this method.

## Further steps
- To further improve the model, I could initialise the snake at a random point in space rather than starting the snake in the top left corner every time. This will allow the agent to see more game states and learn more effectively.
- I could try to use maze solving algorithms to create a more optimal snake agent.