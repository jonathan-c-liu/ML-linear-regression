import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    transformed_features = np.zeros(features.shape)
    for i in range(features.shape[0]):
        transformed_features[i, 0] = np.sqrt(np.square(features[i, 0]) + np.square(features[i, 1]))
        transformed_features[i, 1] = np.arctan(features[i, 1]/features[i, 0])
    return transformed_features

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.weights = None

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        def converged(features, targets, weights):
            wTx = features.transpose() * targets
            wTxy = np.multiply(wTx, targets)
            return True if (wTxy > 0).all() else False

        oneCol = np.ones((features.shape[0], 1))
        features = np.concatenate((oneCol, features), axis=1)
        weights = np.array([1, 1, 1])
        count = 0
        while not converged(features, targets, weights) and count < self.max_iterations:
            count += 1
            for i in range(features.shape[0]):
                if np.dot(weights, features[i, :]) * targets[i] <= 0:
                    weights = weights + features[i, :] * targets[i]

        self.weights = weights

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        preds = np.zeros((features.shape[0]))
        oneCol = np.ones((features.shape[0], 1))
        features = np.concatenate((oneCol, features), axis=1)
        for i in range(features.shape[0]):
            preds[i] = 1 if np.dot(self.weights, features[i, :]) > 0 else -1
        return preds

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        raise NotImplementedError()
