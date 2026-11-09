from abc import ABCMeta, abstractmethod
 
import numpy as np
 
 
class Optimizer(metaclass=ABCMeta):
    """
    Abstract base class for the optimizers of a neural network.
    An optimizer adjusts the parameters of the model during the training, to minimize the loss.
 
    Parameters
    ----------
    learning_rate: float
        The learning rate of the optimizer
    """
 
    def __init__(self, learning_rate: float):
        """
        Initialize the optimizer.
 
        Parameters
        ----------
        learning_rate: float
            The learning rate of the optimizer
        """
        self.learning_rate = learning_rate
 
    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Updates the parameters (weights or biases) of a layer.
        Abstract method that needs to be implemented by all subclasses.
 
        Parameters
        ----------
        w: np.ndarray
            The current parameters of the layer
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters
 
        Returns
        -------
        w: np.ndarray
            The updated parameters
        """
        raise NotImplementedError
 
 
class SGD(Optimizer):
    """
    The Stochastic Gradient Descent (SGD) optimizer, with momentum.
    The momentum accelerates the convergence by accumulating the past gradients.
 
    Parameters
    ----------
    learning_rate: float
        The learning rate used to update the parameters
    momentum: float
        The momentum used to update the parameters
 
    Attributes
    ----------
    retained_gradient: np.ndarray
        The gradient accumulated from the previous updates
    """
 
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.90):
        """
        Initialize the SGD optimizer.
 
        Parameters
        ----------
        learning_rate: float
            The learning rate used to update the parameters
        momentum: float
            The momentum used to update the parameters
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None
 
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Updates the retained gradient, V = momentum * V + (1 - momentum) * dL/dW,
        and the parameters, W = W - learning_rate * V.
 
        Parameters
        ----------
        w: np.ndarray
            The current parameters of the layer
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters
 
        Returns
        -------
        w: np.ndarray
            The updated parameters
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
 
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
 
 
class Adam(Optimizer):
    """
    The Adam optimizer.
    It combines RMSprop and SGD with momentum: it uses the squared gradients to scale the learning rate
    and a moving average of the gradient instead of the gradient itself.
 
    Parameters
    ----------
    learning_rate: float
        The learning rate used to update the parameters
    beta_1: float
        The exponential decay rate for the first moment estimates
    beta_2: float
        The exponential decay rate for the second moment estimates
    epsilon: float
        A small constant for numerical stability
 
    Attributes
    ----------
    m: np.ndarray
        The moving average of the gradient
    v: np.ndarray
        The moving average of the squared gradient
    t: int
        The time stamp, the number of updates done
    """
 
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.
 
        Parameters
        ----------
        learning_rate: float
            The learning rate used to update the parameters
        beta_1: float
            The exponential decay rate for the first moment estimates
        beta_2: float
            The exponential decay rate for the second moment estimates
        epsilon: float
            A small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
 
        self.m = None
        self.v = None
        self.t = 0
 
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Updates the time stamp, the moving averages m and v, and the parameters:
        W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon), where m_hat and v_hat are the
        moving averages corrected for their initialization at zero.
 
        Parameters
        ----------
        w: np.ndarray
            The current parameters of the layer
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters
 
        Returns
        -------
        w: np.ndarray
            The updated parameters
        """
        if self.m is None:
            self.m = np.zeros(np.shape(w))
            self.v = np.zeros(np.shape(w))
 
        self.t += 1
 
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad_loss_w, 2)
 
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
 
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)