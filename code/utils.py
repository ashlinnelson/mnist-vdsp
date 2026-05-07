import numpy as np
import matplotlib.pyplot as plt

def get_2d_weights(synapse_group, n_input, n_e):
    """extracts weights into a 2D matrix"""
    weight_matrix = np.zeros((n_input, n_e))
    weight_matrix[synapse_group.i, synapse_group.j] = synapse_group.w
    return weight_matrix

def plot_weights(weights, n_e):
    """Plots the weights as a grid of 28x28 images."""
    fig, axes = plt.subplots(int(np.sqrt(n_e)), int(np.sqrt(n_e)), figsize=(10, 10))
    wmax = np.max(weights) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[:, i].reshape(28, 28), cmap='hot_r', vmin=0, vmax=wmax)
        ax.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
