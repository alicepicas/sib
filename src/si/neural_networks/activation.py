from abc import abstractmethod
 
import numpy as np
 
from si.neural_networks.layers import Layer
from si.statistics.sigmoid_function import sigmoid_function
 
 
class ActivationLayer(Layer):
    """
    Abstract base class for the activation layers of a neural network.
    Activation layers introduce non-linearity into the model, which allows the network
    to learn complex patterns. Each output depends only on its own input.
 
    Attributes
    ----------
    input: np.ndarray
        The input of the layer
    output: np.ndarray
        The output of the layer
    """
 
    def __init__(self):
        """
        Initialize the activation layer.
        """
        self.input = None
        self.output = None
 
    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Performs forward propagation: applies the activation function to the input.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        training: bool
            Whether the layer is in training mode or in inference mode
 
        Returns
        -------
        output: np.ndarray
            The output of the layer
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output
 
    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Performs backward propagation: multiplies, element by element, the derivative of the
        activation function on the input by the received output error.
 
        Parameters
        ----------
        output_error: np.ndarray
            The error of the output of the layer
 
        Returns
        -------
        input_error: np.ndarray
            The error of the input of the layer
        """
        return self.derivative(self.input) * output_error
 
    @abstractmethod
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the activation function on the given input.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The activation function applied to the input
        """
        raise NotImplementedError
 
    @abstractmethod
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function on the given input.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the activation function on the input
        """
        raise NotImplementedError
 
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer, which is the shape of its input.
 
        Returns
        -------
        shape: tuple
            The shape of the output of the layer
        """
        return self._input_shape
 
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer. Activation layers have no learnable parameters.
 
        Returns
        -------
        parameters: int
            Always 0
        """
        return 0
 
 
class ReLUActivation(ActivationLayer):
    """
    The ReLU (Rectified Linear Unit) activation layer.
    It outputs the input directly if it is positive, and zero otherwise.
    """
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU function: 0 if x < 0, otherwise x.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The ReLU function applied to the input
        """
        return np.where(input >= 0, input, 0)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the derivative of the ReLU function: 0 if x < 0, otherwise 1.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the ReLU function on the input
        """
        return np.where(input >= 0, 1, 0)
 
 
class LeakyReLUActivation(ActivationLayer):
    """
    The Leaky ReLU activation layer.
    It is a variant of the ReLU that, instead of outputting zero for negative inputs,
    outputs a small fraction (alpha) of the input. This keeps a small gradient for
    negative inputs, so the neurons do not stop learning.
 
    Parameters
    ----------
    alpha: float
        The slope of the function for negative inputs
    """
 
    def __init__(self, alpha: float = 0.01):
        """
        Initialize the Leaky ReLU activation layer.
 
        Parameters
        ----------
        alpha: float
            The slope of the function for negative inputs
        """
        super().__init__()
        self.alpha = alpha
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the Leaky ReLU function: alpha * x if x < 0, otherwise x.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The Leaky ReLU function applied to the input
        """
        return np.where(input >= 0, input, self.alpha * input)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the derivative of the Leaky ReLU function: alpha if x < 0, otherwise 1.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the Leaky ReLU function on the input
        """
        return np.where(input >= 0, 1, self.alpha)
 
 
class SigmoidActivation(ActivationLayer):
    """
    The sigmoid activation layer.
    It compresses the input values to the range between 0 and 1.
    """
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function: 1 / (1 + e^-x).
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The sigmoid function applied to the input
        """
        return sigmoid_function(input)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the derivative of the sigmoid function: f(x) * (1 - f(x)).
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the sigmoid function on the input
        """
        sigmoid = self.activation_function(input)
        return sigmoid * (1 - sigmoid)
 
 
class TanhActivation(ActivationLayer):
    """
    The hyperbolic tangent (tanh) activation layer.
    It compresses the input values to the range between -1 and 1.
    """
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the hyperbolic tangent: (e^x - e^-x) / (e^x + e^-x).
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The hyperbolic tangent applied to the input
        """
        return np.tanh(input)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the derivative of the hyperbolic tangent: 1 - f(x)^2.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the hyperbolic tangent on the input
        """
        return 1 - self.activation_function(input) ** 2
 
 
class SoftmaxActivation(ActivationLayer):
    """
    The softmax activation layer.
    It transforms the raw scores into a probability distribution (the values sum to 1),
    which makes it suitable for multi-class classification.
    """
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function: e^x / sum(e^x), along the last axis.
        To avoid very large exponentials, the maximum value is subtracted from the input first.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        output: np.ndarray
            The softmax function applied to the input
        """
        exponentials = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the derivative of the softmax function as given in the slides: f(x) * (1 - f(x)).
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
 
        Returns
        -------
        derivative: np.ndarray
            The derivative of the softmax function on the input
        """
        softmax = self.activation_function(input)
        return softmax * (1 - softmax)