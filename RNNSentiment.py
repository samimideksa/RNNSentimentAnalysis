from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import argparse
#Set the vocabulary size and load in training and test data


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--vocab_size", default=5000, type=int)
    parser.add_argument("--max_words", default=500, type=int)
    parser.add_argument("--embedding_size", default=32, type=int)
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument("--steps",default=200,type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_cmd_args()
    #set the vocabulary size and load and test the training data
    vocabulary_size = args.vocab_size
    max_words = args.max_words
    embedding_size = args.embedding_size
    batch_size = args.batch_size
    num_epochs = args.epochs

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    print('Loaded dataset with {} training samples, {} test samples'.format(len(x_train), len(x_test)))

    print('---review---')
    print(x_train[6])
    print('---label---')
    print(y_train[6])

    #use the dictionary returned by imdb.get_word_index() to map review back to the original words
    word2id = imdb.get_word_index()
    id2word = {i: word for word, i in word2id.items()}
    print('---review with words---')
    print([id2word.get(i, ' ') for i in x_train[6]])
    print('---label---')
    print(y_train[6])

    #maximum review length and minimum review length
    print('Maximum review length: {}'.format(len(max((x_train + x_test), key=len))))

    #print minimum review length
    print('Minimum review length: {}'.format(len(min((x_test + x_test), key=len))))

    #set max words to 500
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)
    
    #keras model with LSTM layer
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    #Evaluate and train model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #specify the batch size and the number of epochs

    x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
    x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

    model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    #scores[1] will correspond to accuracy is we pass metrics=['accuracy]

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    main()
