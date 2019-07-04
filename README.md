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
    
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model


# The Layers can be broken down into 5 different parts:

###### Input Layer (Encoder and Decoder):
###### Embedding Layer (Encoder and Decoder)
###### LSTM Layer (Encoder and Decoder)
###### Decoder Output Layer

 
 # Procedure of summarization:
## Text Preprocessing
 ###### Loading Story:
 ```
 def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        #doc = load_doc(filename)
        #story, highlights = split_story(doc)
        story = load_doc(filename)
        if (len(story) >= 5):
            stories.append({'story':story})#, 'highlights':highlights, 'summary':''})
    return stories
directory = 'dataset/stories_text_summarization_dataset_test'
stories = load_stories(directory)
```
 ###### Splititng story:
 ```
 def split_story(doc):
    index = doc.find('@highlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights
   ```
  ###### cleaning lines:
  ```
  def clean_lines(lines):
    cleaned = list()
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        line = line.split()
        line = [word.lower() for word in line]
        line = [w.translate(table) for w in line]
        line = [word for word in line if word.isalpha() and not word in stop_words]
        cleaned.append(' '.join(line))
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned
  ```

  
## Put BOS tag and EOS tag for decoder input
  means “Begin of Sequence”, and “End of Sequence”.
 ``` 
  def tagger(decoder_input_sentence):
  bos = "<BOS> "
  eos = " <EOS>"
  final_target = [bos + text + eos for text in decoder_input_sentence] 
  return final_target

decoder_inputs = tagger(decoder_input_text)
 ```
## Vocabulary 
  
 ``` from keras.preprocessing.text import Tokenizer

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
```
## Tokenize Bag of words to Bag of IDs
```
from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999

def text2seq(encoder_text, decoder_text, VOCAB_SIZE):

  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
  decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
  
  return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(encoder_text, decoder_text, VOCAB_SIZE) 
```

## Padding
```
from keras.preprocessing.sequence import pad_sequences

def padding(encoder_sequences, decoder_sequences, MAX_LEN):
  
  encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  
  return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN):
```
## Word Embedding
 We use Pretraind Word2Vec Model from Glove
 ```
import numpy as np
word_embeddings = {}
f = open(r'glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    #print(values)
    word = values[0]
    #print(word)
    coefs = np.asarray(values[1:], dtype='float32')
    #print(coefs)
    word_embeddings[word] = coefs
f.close()
 ```
## Reshape the Data depends on neural network shape

## Split Data for training and validation, testing
```
 from sklearn.model_selection import train_test_split
```
# For Example:
## Input:
Australian wine exports hit a record 52.1 million litters worth 260 million dollars (143 million us) in September, the government statistics office reported on Monday
## Output:
Australian wine exports hit record high in September

# 
The encoder-decoder model is composed of encoder and decoder like its name.The encoder converts an input document to a latent representation (vector), and the decoder generates a summary by using it.
