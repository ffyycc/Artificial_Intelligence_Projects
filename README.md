# artificial-intelligence-project
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
