# Abstractive-summarization

As like the machine translation model converts a source language text to a target one, the summarization system converts a source document to a target summary.

Nowadays, encoder-decoder model that is one of the neural network models is mainly used in machine translation. So this model is also widely used in abstractive summarization model.

encoder-decoder summarization model, tensorflow offers basic model: IT turns out for shorter texts, summarization can be learned end-to-end with a deep learning technique called sequence-to-sequence learning.

End-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multi-layered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector.

 # Seq2Seq model arch:
def seq2seq_model_builder(HIDDEN_DIM=300):
    
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model


# The Layers can be broken down into 5 different parts:

# Input Layer (Encoder and Decoder):
encoder_input_layer = Input(shape=(sequence_length, ))
decoder_input_layer = Input(shape=(sequence_length, ))

# Embedding Layer (Encoder and Decoder)
 embedding_layer = Embedding(input_dim = vocab_size,
                            output_dim = embedding_dimension, 
                            input_length = sequence_length)
# LSTM Layer (Encoder and Decoder)
encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)   
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

# Decoder Output Layer
 
 outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
 
 
 # Whole process can be:
Text CLeaning
Put <BOS> tag and <EOS> tag for decoder input
  <BOS> means “Begin of Sequence”, <EOS> means “End of Sequence”.
Make Vocabulary (VOCAB_SIZE)
    from keras.preprocessing.text import Tokenizer

def vocab_creater(text_lists, VOCAB_SIZE):

  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  tokenizer.fit_on_texts(text_lists)
  dictionary = tokenizer.word_index
  
  word2idx = {}
  idx2word = {}
  for k, v in dictionary.items():
      if v < VOCAB_SIZE:
          word2idx[k] = v
          index2word[v] = k
      if v >= VOCAB_SIZE-1:
          continue
          
  return word2idx, idx2word

word2idx, idx2word = vocab_creater(text_lists=encoder_input_text+decoder_input_t

Tokenize Bag of words to Bag of IDs
from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999

def text2seq(encoder_text, decoder_text, VOCAB_SIZE):

  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
  decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
  
  return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(encoder_text, decoder_text, VOCAB_SIZE) 


Padding (MAX_LEN)

from keras.preprocessing.sequence import pad_sequences

def padding(encoder_sequences, decoder_sequences, MAX_LEN):
  
  encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  
  return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN):

Word Embedding (EMBEDDING_DIM)
 We use Pretraind Word2Vec Model from Glove
 
Reshape the Data depends on neural network shape

Split Data for training and validation, testing

# For Example:
# Input:
Australian wine exports hit a record 52.1 million litters worth 260 million dollars (143 million us) in September, the government statistics office reported on Monday
# Output:
Australian wine exports hit record high in September

# 
The encoder-decoder model is composed of encoder and decoder like its name.The encoder converts an input document to a latent representation (vector), and the decoder generates a summary by using it.
