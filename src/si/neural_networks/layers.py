import copy
from abc import ABCMeta, abstractmethod
 
import numpy as np
 
 
class Layer(metaclass=ABCMeta):
    """
    Abstract base class for the layers of a neural network.
    A layer receives an input, transforms it and produces an output.
    """
 
    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Computes the output of the layer for a given input.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer
        training: bool
            Whether the network is being trained or not
 
        Returns
        -------
        output: np.ndarray
            The output of the layer
        """
        raise NotImplementedError
 
    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.
        Abstract method that needs to be implemented by all subclasses.
 
        Returns
        -------
        shape: tuple
            The shape of the output of the layer
        """
        raise NotImplementedError
 
    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.
        Abstract method that needs to be implemented by all subclasses.
 
        Returns
        -------
        parameters: int
            The number of parameters of the layer
        """
        raise NotImplementedError
 
    def set_input_shape(self, input_shape: tuple):
        """
        Sets the shape of the input of the layer.
 
        Parameters
        ----------
        input_shape: tuple
            The shape of the input of the layer
        """
        self._input_shape = input_shape
 
    def input_shape(self) -> tuple:
        """
        Returns the shape of the input of the layer.
 
        Returns
        -------
        shape: tuple
            The shape of the input of the layer
        """
        return self._input_shape
 
    def layer_name(self) -> str:
        """
        Returns the name of the layer.
 
        Returns
        -------
        name: str
            The name of the layer (the name of its class)
        """
        return self.__class__.__name__
 
 
class DenseLayer(Layer):
    """
    A dense layer, also known as a fully connected layer.
    Each neuron of the layer is connected to every neuron of the previous layer.
 
    Parameters
    ----------
    n_units: int
        The number of neurons of the layer
    input_shape: tuple
        The shape of the input of the layer. For tabular data it is (n_features,)
 
    Attributes
    ----------
    input: np.ndarray
        The input of the layer
    output: np.ndarray
        The output of the layer
    weights: np.ndarray
        The weights of the layer (n_features, n_units)
    biases: np.ndarray
        The biases of the layer (1, n_units)
    """
 
    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.
 
        Parameters
        ----------
        n_units: int
            The number of neurons of the layer
        input_shape: tuple
            The shape of the input of the layer. For tabular data it is (n_features,)
        """
        self.n_units = n_units
        self._input_shape = input_shape
 
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None
 
    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Initializes the weights with random values between -0.5 and 0.5, and the biases with zeros.
        If an optimizer is given, a copy of it is kept for the weights and another for the biases.
 
        Parameters
        ----------
        optimizer
            The optimizer used to update the weights and the biases of the layer
 
        Returns
        -------
        self: DenseLayer
            The initialized layer
        """
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
 
        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)
 
        return self
 
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer (weights and biases).
 
        Returns
        -------
        parameters: int
            The number of parameters of the layer
        """
        return int(np.prod(self.weights.shape) + np.prod(self.biases.shape))
 
    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs forward propagation: the weighted sum of the input plus the biases, Y = XW + B.
 
        Parameters
        ----------
        input: np.ndarray
            The input of the layer (n_samples, n_features)
        training: bool
            Whether the network is being trained or not
 
        Returns
        -------
        output: np.ndarray
            The output of the layer (n_samples, n_units)
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.
 
        Returns
        -------
        shape: tuple
            The shape of the output of the layer, (n_units,)
        """
        return (self.n_units,)
    