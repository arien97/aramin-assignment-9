import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

# Create the result directory
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Select activation function and its derivative
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        out = sigmoid(self.z2)
        
        # Store activations for visualization
        self.activations = (X, self.a1, out)
        return out

    def backward(self, X, y):
        m = y.shape[0]
        
        # Output layer gradients
        dz2 = (self.activations[2] - y) / m
        dW2 = np.dot(self.activations[1].T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        # Store gradients for visualization
        self.gradients = (dW1, dW2)

# Generate data (circular decision boundary)
def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):  # Perform 10 training steps per frame to speed up the process
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden layer features with two colors (red and blue)
    hidden_features = mlp.activations[1]
    colors = ['red' if label == -1 else 'blue' for label in y.ravel()]
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=colors, alpha=0.7)
    
    step_num = frame * 10  # Step number for visualization
    ax_hidden.set_title(f'Hidden Space at Step {step_num}')
    
    # Plot decision boundary in the hidden space
    x_min, x_max = hidden_features[:, 0].min() - 0.5, hidden_features[:, 0].max() + 0.5
    y_min, y_max = hidden_features[:, 1].min() - 0.5, hidden_features[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / mlp.W2[2]
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.2)

    # Input space visualization with changing decision boundary based on steps and background colors based on boundaries
    x_min_input, x_max_input = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min_input, y_max_input = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx_input, yy_input = np.meshgrid(np.linspace(x_min_input, x_max_input, 100), np.linspace(y_min_input, y_max_input, 100))
    grid_input = np.c_[xx_input.ravel(), yy_input.ravel()]
    Z_input = mlp.forward(grid_input)
    Z_input = Z_input.reshape(xx_input.shape)
    
    ax_input.contourf(xx_input, yy_input, Z_input, alpha=0.3, cmap='bwr')
    ax_input.contour(xx_input, yy_input, Z_input, levels=[0], colors='k')
    ax_input.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
    
    ax_input.set_title(f'Input Space at Step {step_num}')

    # Gradient visualization with edge thickness representing magnitude of gradient and spread out circles
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            grad_magnitude = np.abs(mlp.gradients[0][i,j])
            ax_gradient.plot([i*3, j*3+3], [0, 1], 'gray', linewidth=grad_magnitude*50)

    for i in range(mlp.W2.shape[0]):
        grad_magnitude = np.abs(mlp.gradients[1][i])
        ax_gradient.plot([i*3+3, 6], [1, 2], 'gray', linewidth=grad_magnitude*50)

    for i in range(mlp.W1.shape[0]):
        ax_gradient.add_patch(Circle((i*3, 0), 0.2, color='lightblue', transform=ax_gradient.transData))

    for i in range(mlp.W2.shape[0]):
        ax_gradient.add_patch(Circle((i*3+3, 1), 0.2, color='lightgreen', transform=ax_gradient.transData))
    ax_gradient.add_patch(Circle((6, 2), 0.2, color='lightcoral', transform=ax_gradient.transData))

    ax_gradient.set_xlim(-1, 10)
    ax_gradient.set_ylim(-0.5, 2.5)
    ax_gradient.set_aspect('equal')
    ax_gradient.axis('off')
    
    ax_gradient.set_title('Network Architecture')

# Function to visualize training progress
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, 
                                   ax_gradient=ax_gradient, X=X, y=y), 
                       frames=step_num // 10, repeat=False)
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

# Visualize for all activations
def visualize_all():
    activations = ['tanh', 'relu', 'sigmoid']
    lr = 0.1
    step_num = 1000
    
    for activation in activations:
        visualize(activation, lr, step_num)

if __name__ == "__main__":
    visualize_all()