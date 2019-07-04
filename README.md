# Abstractive-summarization

As like the machine translation model converts a source language text to a target one, the summarization system converts a source document to a target summary.

Nowadays, encoder-decoder model that is one of the neural network models is mainly used in machine translation. So this model is also widely used in abstractive summarization model.

encoder-decoder summarization model, tensorflow offers basic model: IT turns out for shorter texts, summarization can be learned end-to-end with a deep learning technique called sequence-to-sequence learning.

End-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multi-layered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector.

# For Example:
# Input:
Australian wine exports hit a record 52.1 million litters worth 260 million dollars (143 million us) in September, the government statistics office reported on Monday
# Output:
Australian wine exports hit record high in September

# 
The encoder-decoder model is composed of encoder and decoder like its name.The encoder converts an input document to a latent representation (vector), and the decoder generates a summary by using it.
