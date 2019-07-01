import keras
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import layers
from keras.layers import GRU, LSTM, SimpleRNN, Dense
import keras
import numpy as np
import random
import sys

print(keras.__version__)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# List of unique characters in the corpus
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
print('CHARS:\t', chars)
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)
print('char_indices:\t', char_indices)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(GRU(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

for epoch in range(1, 10):
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(x, y, batch_size=128, epochs=1)
    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# As you can see, a low temperature results in extremely repetitive and predictable text, but where local structure is highly realistic: in 
# particular, all words (a word being a local pattern of characters) are real English words. With higher temperatures, the generated text 
# becomes more interesting, surprising, even creative; it may sometimes invent completely new words that sound somewhat plausible (such as 
# "eterned" or "troveration"). With a high temperature, the local structure starts breaking down and most words look like semi-random strings 
# of characters. Without a doubt, here 0.5 is the most interesting temperature for text generation in this specific setup. Always experiment 
# with multiple sampling strategies! A clever balance between learned structure and randomness is what makes generation interesting.
# 
# Note that by training a bigger model, longer, on more data, you can achieve generated samples that will look much more coherent and 
# realistic than ours. But of course, don't expect to ever generate any meaningful text, other than by random chance: all we are doing is 
# sampling data from a statistical model of which characters come after which characters. Language is a communication channel, and there is 
# a distinction between what communications are about, and the statistical structure of the messages in which communications are encoded. To 
# evidence this distinction, here is a thought experiment: what if human language did a better job at compressing communications, much like 
# our computers do with most of our digital communications? Then language would be no less meaningful, yet it would lack any intrinsic 
# statistical structure, thus making it impossible to learn a language model like we just did.
# 
# 
# ## Take aways
# 
# * We can generate discrete sequence data by training a model to predict the next tokens(s) given previous tokens.
# * In the case of text, such a model is called a "language model" and could be based on either words or characters.
# * Sampling the next token requires balance between adhering to what the model judges likely, and introducing randomness.
# * One way to handle this is the notion of _softmax temperature_. Always experiment with different temperatures to find the "right" one.
