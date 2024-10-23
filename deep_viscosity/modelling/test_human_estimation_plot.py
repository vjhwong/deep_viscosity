import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot_data():
    # Load the data for the line plot
    true_viscosities = np.load('vectors/all_viscosities.npy')
    p1_predictions = np.load('vectors/testperson1_prediction.npy')
    p2_predictions = np.load('vectors/testperson2_prediction.npy')
    p3_predictions = np.load('vectors/testperson3_prediction.npy')
    p4_predictions = np.load('vectors/testperson4_prediction.npy')
    
    # Load the data for the scatter plot
    test_viscosities = np.load('models/expert-wildflower-177/expert-wildflower-177_testtargets.npy')
    test_predictions = np.load('models/expert-wildflower-177/expert-wildflower-177_testpredictions.npy')
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the line
    plt.plot(true_viscosities, true_viscosities, label='Predictions = True Viscosities', color='black', linestyle='dashed')

    # Human predictions
    plt.scatter(true_viscosities, p1_predictions, label='Test Person 1', color='blue', alpha=0.3, edgecolors='none')
    plt.scatter(true_viscosities, p2_predictions, label='Test Person 2', color='green', alpha=0.3, edgecolors='none')
    plt.scatter(true_viscosities, p3_predictions, label='Test Person 3', color='purple', alpha=0.3, edgecolors='none')
    plt.scatter(true_viscosities, p4_predictions, label='Test Person 4', color='orange', alpha=0.3, edgecolors='none')

    # Model predictions
    plt.scatter(test_viscosities, test_predictions, label='3D CNN Model', color='red', alpha = 0.3, edgecolors='none')

    # Add labels and title
    plt.title('Predictions by the Model and by Humans')
    plt.xlabel('True Viscosities')
    plt.ylabel('Predictions')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Call the function to create the plot
plot_data()
