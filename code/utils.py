import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

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

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax)

    # Set up the axes labels
    classes = np.arange(10)
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add axis titles
    ax.set_ylabel('True Digit (Actual)', fontsize=14)
    ax.set_xlabel('Predicted Digit (SNN Output)', fontsize=14)
    ax.set_title('SNN Confusion Matrix', fontsize=16, pad=20)

    # Loop over the matrix data and add the raw numbers as text inside the boxes
    # change the text color to white if the box background is very dark
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=12)

    plt.tight_layout()
    plt.show()

def show_spikes(spike_mon, a=0, b=2000):
	plt.figure(figsize=(14, 5))
	plt.plot(spike_mon.t / ms, spike_mon.i, '.k', markersize=2)
	plt.xlabel('Time (ms)')
	plt.ylabel('Neuron Index')
	plt.title('Excitatory Layer Raster Plot (Who fired and when?)')
	# Limit the view to the first few images
	plt.xlim(a, b) 
	plt.show()
	
	# Total Spikes per Neuron
	plt.figure(figsize=(14, 4))
	spike_counts = np.array(spike_mon.count)
	plt.bar(range(len(spike_counts)), spike_counts, color='royalblue')
	plt.xlabel('Neuron Index')
	plt.ylabel('Total Spikes')
	plt.title('Total Spikes per Neuron (Is Winner-Take-All working?)')
	plt.show()