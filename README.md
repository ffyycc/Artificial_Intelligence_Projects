# Artificial Intelligence Project
Below is the list of my projects on different perspectives of artificial intelligence.
## 1. Search 

**Brief description**: Use BFS, A*, and greedy algorithms to solve 'pacman'-like maze puzzles

Running method:

`python3 main.py --human data/<part-name>/<map-name> --search <algorithm-name>`

Ex:
`python3 main.py data/part-1/small --search bfs`

`python3 main.py data/part-3/corner --search astar_multiple`

`python3 main.py data/part-3/corner --search fast`

## 2. Robotics

**Brief description**:
The project transforms a shapeshifting alien path planning problem into a configuration space <horizontal, vertical, and ball shape>, and then search for a path in that space.

Running method:

`python3 geometry.py`

## 3. Naive Bayes

**Brief description**:
The proect uses Naive Bayes algorithm to train a binary sentiment classifier with a dataset of movie reviews. The output model could evaluate whether the IMDb comments are positive or negative.

Running method:

`python3 mp3.py`

## 4. HMM POS tagging

**Brief description**:

This project implements part of speech (POS) tagging using an HMM model. My model ends up with 96.11% accuracy in test data.

Running method:

`python3 test_viterbi.py`

## 5. Perceptron and kNN

**Brief description**

This project implements the perceptron and K-nearest neighbors algorithms to detect whether or not an image contains an animal or not.

Running method:

`python3 mp5.py -h`

## 6. Neural Nets

This project uses Pytorch to generate multiple layers convolutional neural networks. The classifier divides images into four categories: ship, automobile, dog, or frog. My neural networks have the accuracy about 82.3% on image classification.

Running method:

`python3 mp6.py -h`

## 7. Reinforcement Learning - Snake Game

This project trains the agent in snake game using Q-learning. The snake needs to eat food in a 11x11 game map. However, the snake could not hit its body. My Q-learning strategy using reinforcement learning could eat 23 food on average without knowing where the food is generated.

Running method:

`python mp7.py --human`
