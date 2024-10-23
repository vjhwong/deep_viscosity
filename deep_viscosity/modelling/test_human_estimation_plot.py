import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def plot_data(only_test_visc: bool = False):
    """
    Creates a plot showing the human predictions along with the model predictions on the test set.

    Args:
        only_test_vist (bool): determines if all human predictions should be plottd or only the test viscosities
    """

    # Load the data for the plots
    true_viscosities = np.load("vectors/all_viscosities.npy")
    p1_predictions = np.load("vectors/testperson1_prediction.npy")
    p2_predictions = np.load("vectors/testperson2_prediction.npy")
    p3_predictions = np.load("vectors/testperson3_prediction.npy")
    p4_predictions = np.load("vectors/testperson4_prediction.npy")
    test_viscosities = np.load(
        "models/expert-wildflower-177/expert-wildflower-177_testtargets.npy"
    )
    test_predictions = np.load(
        "models/expert-wildflower-177/expert-wildflower-177_testpredictions.npy"
    )

    if only_test_visc:
        # Find only human predictions for test viscosities
        unique_viscosities = list(
            map(lambda x: round(float(x), 2), np.unique(test_viscosities))
        )
        human_predictions = [
            p1_predictions,
            p2_predictions,
            p3_predictions,
            p4_predictions,
        ]
        desired_human_predictions = [[] for _ in range(4)]
        for human_prediction, desired_human_prediction in zip(
            human_predictions, desired_human_predictions
        ):
            for visc, pred in zip(true_viscosities, human_prediction):
                if visc in unique_viscosities:
                    desired_human_prediction.append(pred)
        true_viscosities = unique_viscosities
        p1_predictions = desired_human_predictions[0]
        p2_predictions = desired_human_predictions[1]
        p3_predictions = desired_human_predictions[2]
        p4_predictions = desired_human_predictions[3]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the predictions = truth line
    plt.plot(
        true_viscosities,
        true_viscosities,
        label="Predictions = True Viscosities",
        color="black",
        linestyle="dashed",
        alpha=0.5,
    )

    # Human predictions
    plt.scatter(
        true_viscosities,
        p1_predictions,
        label="Test Person 1",
        color="blue",
        alpha=0.3,
        edgecolors="none",
    )
    plt.scatter(
        true_viscosities,
        p2_predictions,
        label="Test Person 2",
        color="green",
        alpha=0.3,
        edgecolors="none",
    )
    plt.scatter(
        true_viscosities,
        p3_predictions,
        label="Test Person 3",
        color="purple",
        alpha=0.3,
        edgecolors="none",
    )
    plt.scatter(
        true_viscosities,
        p4_predictions,
        label="Test Person 4",
        color="orange",
        alpha=0.3,
        edgecolors="none",
    )

    # Model predictions
    plt.scatter(
        test_viscosities,
        test_predictions,
        label="3D CNN Model",
        color="red",
        alpha=0.5,
        edgecolors="none",
    )

    # Add labels and title
    plt.title("Predictions by the Model and by Humans", fontsize=16)
    plt.xlabel("True Viscosities", fontsize=14)
    plt.ylabel("Predictions", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")

    # Show the plot
    plt.grid(True)
    plt.show()


def main():
    plot_data(only_test_visc=True)


if __name__ == "__main__":
    main()
