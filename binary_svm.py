import numpy as np
from matplotlib import pyplot as plt


""" Training using Stochastic Gradient Descent """
def train_svm(X, Y):
    """
    :param X: Trainig Data Points
    :param Y: Correct Labes For the Training Points
    :return: Weights after minimising the error
    """
    epochs = 100000
    # Learning Rate
    eta = 1
    # Randomly initialized Weights
    W = np.zeros(len(X[0]))
    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], W)) < 1:
                error = 1
                W = W + eta * ((Y[i] * X[i]) + (-2 * (1/epoch) * W))
            else:
                W = W + eta * (-2 * (1/epoch) * W)
            errors.append(error)
    # Plots the rate of classification errors during training for our SVM
    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return W


if __name__ == '__main__':
    X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],
    ])

    Y = np.array([-1, -1, 1, 1, 1])
    # Plots training examples on a 2D graph!
    for d, sample in enumerate(X):
        # Plot the negative samples (the first 3)
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        # Plot the positive samples (the last 2)
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

    w = train_svm(X, Y)
    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        # Plot the positive samples
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

    # Add our test samples
    plt.scatter(2, 2, s=120, marker='_', linewidths=2, color='yellow')
    plt.scatter(4, 3, s=120, marker='+', linewidths=2, color='blue')

    # Print the hyperplane calculated by train_svm()
    x2 = [w[0], w[1], -w[1], w[0]]
    x3 = [w[0], w[1], w[1], -w[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')
    plt.show()

