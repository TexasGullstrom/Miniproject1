import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load("/Desktop/BV2/Hw2/HandwrittenDigits/TrainDigits.npy")
TrainLab = np.load("/Desktop/BV2/Hw2/HandwrittenDigits/TrainLabels.npy")

def get_matrix_dimensions(matrix):
    """
    Returns the dimensions of a given matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix.

    Returns:
    tuple: A tuple representing the dimensions of the matrix (rows, columns).
    """
    if isinstance(matrix, np.ndarray):
        return matrix.shape
    else:
        raise ValueError("Input must be a NumPy array.")


index3= (TrainLab==3)
A3 = TrainMat[:,index3[0]] # The first digit in the training set
A3=A3[:,0:400]
[U,S,Vt]=np.linalg.svd(A3,full_matrices='False')

D = np.reshape(A3[:,1], (28, 28)).T # Reshaping a vector to a matrix
plt.imshow(D, cmap ='gray') # Plot of the digit
plt.title("Digit '3'")
plt.axis('off')  # Hide the axis
plt.show()  # Display the plot


S[-1:]=0
P=np.zeros((784,400))
S=np.eye(400)@S

# Create a new matrix of zeros with shape (784, 400)
S_extended = np.zeros((784, 400))

# Insert the original matrix S into the first 400 rows of S_extended
S_extended[:400, :] = S

print(get_matrix_dimensions(U),get_matrix_dimensions(S),get_matrix_dimensions(Vt))

A3=U@S_extended@Vt

D = np.reshape(A3[:,1], (28, 28)).T # Reshaping a vector to a matrix
plt.imshow(D, cmap ='gray') # Plot of the digit
plt.title("Digit '3'")
plt.axis('off')  # Hide the axis
plt.show()  # Display the plot