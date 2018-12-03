# RNNSentimentAnalysis
This project uses Recurrent Neural Networks for Sentiment Analysis as a better approach. Its paper can be found in this [link](https://arxiv.org/ftp/arxiv/papers/1801/1801.07883.pdf)
We will train a classfier movie reviews in [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/), using [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network).
### Prerequisites
```
keras
tensorflow
``` 
you can create a conda virtual environment to run the project by using the following command 
```
conda env create -f environment.yml
```
### Dataset 
We will use Recurrent Neural Networks, and in particular [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory), to perform sentiment analysis in [Keras](https://keras.io/). Conveniently, Keras has a built-in IMDb movie reviews data set that we can use.
```
from keras.daasets import imdb
```
## How to train the model
To start training program use this command
```
python RNNSentiment.py --epochs epochs --batch_size batch_size --vocab_size vocabulary_size --max_words maximum_words --embedding_size embedding_size --lr learning_rate --steps steps_per_epoch
```
where 
* --epochs => no of epochs to train the model
* --batch_size => batch size 
* --vocab_size => vocabulary size
* --max_words => the maximum number of words
* --embedding_size => word embedding size
* --lr => learning rate 
* --steps => steps per epoch 