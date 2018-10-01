"""Keras, History functions.
"""

import numpy as np
import matplotlib.pyplot as plt

def save_history(history, result_file):
    """Save history instance as file.

    # Arguments
        history: Returns of fit method.
        result_file: Path to save as text file. 

    # Returns
        Save as file.
    """

    metrics = list(history.history.keys())
    epochs = len(history.history[metrics[0]])
    num_metrics = len(metrics)

    with open(result_file, 'w') as txt:
        txt.write('epoch\t' + '\t'.join([str(a) for a in metrics]) + '\n')
        for i in range(epochs):
            txt.write(('%d\t' + '\t'.join(['%f' for num in range(num_metrics)]) + '\n') % (tuple([i]) + tuple(history[key][i] for key in metrics)))

def get_array(files):
    """Convert file to numpy array.

    # Arguments
        files: Path to file, saved by above save_history method.

    # Returns
        labels: Dictionary, Keys(file_path) and Values(metrics name).
        values: Dictionary, Keys(file_path) and Values(metrics value).
    """

    labels, values = {}, {}
    for file in files:
        with open(file, 'r') as txt:
            lines = txt.read()
        contents = []
        labels[file] = lines.split('\n')[0].split('\t')
        for line in lines.split('\n')[1:-1]:
            contents.append(list(map(float, line.split('\t'))))
        values[file] = np.array(contents)
    
    return labels, values

def show_history(metrics='acc', average=False, *files):
    """Show history.

    # Arguments
        metrics: Metrics name. If 'acc', you can see 'acc' and 'val_acc'.
        average: Moving average. (e.g. 3 and 5)
        files: Path to file, saved by above save_history method. It receives multiple files.

    # Returns
        Show as integrated graph.
    """

    labels, values = get_array(files)
    colors = ["b", "g", "r", "c", "m", "y", "b", "w"]
    plt.figure(figsize=(12, 8))
    for i, key in enumerate(values.keys()):
        if average:
            for column in range(1, values[key].shape[1]):
                values[key][:, column] = np.convolve(values[key][:, column], np.ones(average)/float(average), 'same')
                values[key] = values[key][average//2:-((average//2)+1)]
        plt.plot(values[key][:, 0], values[key][:, labels[key].index(metrics)], colors[i], alpha=0.3, label=key[:-4]+' '+metrics)
        plt.plot(values[key][:, 0], values[key][:, labels[key].index('val_'+metrics)], colors[i], alpha=0.9, label=key[:-4]+' '+'val_'+metrics)

    plt.title('Training and validation history')
    plt.xlabel('Epochs')
    plt.ylabel(metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.00), fontsize=12)
    plt.grid(color='gray', alpha=0.3)
    plt.show()