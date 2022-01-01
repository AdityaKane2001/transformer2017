import tensorflow as tf
from tensorflow.keras import layers

MAX_SEQ_LEN = 64
# BATCH_SIZE = 25000
DROPOUT_RATE = 0.1
EMBEDDING_DIMS = 512
VOCABULARY_SIZE = 4096
N_TRANSFORMERS = 6
FFNN_DIMS = 2048
NUM_HEADS = 8
KEY_DIMS = EMBEDDING_DIMS/ NUM_HEADS
VALUE_DIMS = EMBEDDING_DIMS/ NUM_HEADS

"""
Input pipeline:
1. We get batch_size number of pairs of sentences from the dataset: 
    batch_size x  ("My name is Aditya Kane", "<start> Ich bin Aditya Kane <end>")
2. These sentences are then tokenized: 
    batch_size x  ([2,3,4,5,6],[1,15,7,8,9,1000])
3. The sentences are then padded to the largest sentence: 
    batch_size x ([2,3,4,5,6,0,0,0,0], [1,15,7,8,9,1000,0,0,0])
4. They are then converted to embeddings:
    batch_size x max_seq_len x embedding_dims
5. Add positional embeddings to this
    batch_size x max_seq_len x embedding_dims
This is the input to our model.
"""

class PositionAwareEmbeddings(layers.Layer):
    def __init__(self):
        self.dropout = layers.Dropout(DROPOUT_RATE)
        self.embed_dims = EMBEDDING_DIMS
        self.vocab_size = VOCABULARY_SIZE
        self.embeddings = layers.Embedding(VOCABULARY_SIZE, EMBEDDING_DIMS,
                                    input_length=MAX_SEQ_LEN)
        self.max_seq_len = MAX_SEQ_LEN
        

    def get_positional_embeddings(self, input_seq_len):
        """
        The basic idea is that we use sin(frequency x timestep) for even tsteps
        and cos(frequency x timestep) for odd timesteps. The frequency follows a 
        GP, which results in something like binary (00, 01, 10, 11). Thus, 
        effectively we are taking a continuous version of these binary positional
        embeddings in the form of sinusoidal waves.

        The logic is a bit messy, which is because location update is not 
        allowed in TF.

        positions = [0,1,..., 99]
        freqs = [1/(10000^i/512) for all even i in (0,512)]
        We cross multiply these two, which gives us frequency map for all 
        embedding dimensions to timestep (seq_len x emb_dims)

        We then take sin/cos function of this map and return it. This will be 
        added to the embedding. Note that all embeddings and each value in the 
        embeddings is affected differently. All values froma single embedding 
        are not treated the same.  

        Also note positional embeddings need not consider the batch size as the 
        batch_size is broadcasted. 
        """
        positions = tf.reshape(tf.range(input_seq_len, dtype=tf.double), (input_seq_len,1))
        freqs = tf.math.pow(10000, 
                -tf.range(0, self.embed_dims, delta=2) / self.embed_dims)

        sin_embs = tf.transpose(tf.cast(tf.math.sin(positions * freqs), tf.float32))
        cos_embs = tf.transpose(tf.cast(tf.math.cos( positions* freqs), tf.float32))

        # A brief explaination for scatter_nd. It is basically a function for
        # inserting some values in a zero matrix at certain positions. 
        #
        # `indices` must have the indices in which the individual slices of 
        # `updates` must be inserted. Thus, the shape of the slice should match 
        # the dimension to be inserted. `shape` is the final output shape.
        #
        # Following is a small example whihc will make things clear.
        #
        # ```
        # a = tf.constant(tf.random.uniform((4,5)))
        # a_ = tf.scatter_nd( indices = [[0], [2], [3], [4] ]  ,
        #             updates = a,
        #             shape = [6,5])
        # ``` 
        # Output: 
        # <tf.Tensor: shape=(6, 5), dtype=float32, numpy=
        # array([[0.4543445 , 0.88263774, 0.1650244 , 0.20423841, 0.995103  ],
        #     [0.        , 0.        , 0.        , 0.        , 0.        ],
        #     [0.8253639 , 0.05689311, 0.02443647, 0.31397212, 0.8881415 ],
        #     [0.8112322 , 0.52572775, 0.23904228, 0.7718793 , 0.48822534],
        #     [0.26956844, 0.69462633, 0.46506262, 0.42149532, 0.44453228],
        #     [0.        , 0.        , 0.        , 0.        , 0.        ]],
        #     dtype=float32)>

        expanded_sin_embs = tf.scatter_nd( 
            indices = [[i] for i in range(512) if i%2==1],
            updates = sin_embs,
            shape = ( self.embed_dims, input_seq_len)
        )
        expanded_cos_embs = tf.scatter_nd( 
            indices = [[i] for i in range(512) if i%2==0], 
            updates = cos_embs,
            shape = ( self.embed_dims, input_seq_len)
        )
        pos_embs = tf.transpose(expanded_sin_embs + expanded_cos_embs)
        return pos_embs, expanded_sin_embs,expanded_cos_embs

    def call(self, inputs):
        input_seq_len = inputs.shape[-1]
        pos_emb = self.get_positional_embeddings(input_seq_len)
        outputs = self.embeddings(inputs)
        outputs += pos_emb

        return outputs