from tensorflow import keras
import os

from model import create_vivit_classifier
from embedding import TubeletEmbedding, PositionalEncoder
from preprocessing.loader import create_dataloaders


def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Train the model.
    _ = model.fit(trainloader, epochs=20, validation_data=validloader)

    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model

processed_data_path = os.path.join("data", "processed")
trainloader, testloader, validloader = create_dataloaders(
    batch_size=16,
    processed_data_path=processed_data_path,
    validation_size=0.15,
    test_size=0.15,
)


LEARNING_RATE = 1e-4
PATCH_SIZE = (8, 8, 8)
PROJECTION_DIM = 128

model = run_experiment()