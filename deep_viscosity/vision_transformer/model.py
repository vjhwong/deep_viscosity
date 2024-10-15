import tensorflow as tf
from tensorflow import keras
from keras import layers


INPUT_SHAPE = (1, 55, 210, 220)
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
):
    # Define the input layer for the model
    inputs = layers.Input(shape=input_shape)
    
    # Create patches from the input data using the tubelet_embedder
    patches = tubelet_embedder(inputs)
    
    # Encode the patches using positional_encoder
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization and Multi-Head Self-Attention (MHSA)
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Add a skip connection to the output of MHSA
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and Multi-Layer Perceptron (MLP)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Add another skip connection, connecting to the output of the MLP
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify the outputs using a dense layer with softmax activation
    outputs = layers.Dense(units=1, activation="softmax")(representation)

    # Create the Keras model, connecting the input and output layers
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model