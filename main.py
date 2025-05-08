import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load("TrainDigits.npy")
TrainLab = np.load("TrainLabels.npy")

TestDig = np.load("TestDigits.npy")
TestLab = np.load("TestLabels.npy")

#print(TestLab[0,7]) # printar labels

#siffran 3
index = (TrainLab == 3); # find train digits of type 3
A3 = TrainMat[:,index[0]] # all train digits of type 3
A3 = A3[:,0:400] # the first 400 train digits of type 3

U3,S3,Vt3 = np.linalg.svd(A3) # SVD


A1= np.dot(U3[:,:1] * S3[:1], Vt3[:1,:])

if False: # Visar den perfekta 3:an
    D3 = np.reshape(A1[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D3, cmap ="gray") # Den perfekta 3:an
    plt.show()
 


d = TestDig[:,8] # en testbild
if False:
    D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D, cmap ="gray") # Plot of the digit
    plt.show()


#x = np.linalg.solve(U,A3)

# Beräkna residual



# beräkna residual, exempel 
r = np.linalg.norm((np.eye(np.shape(U3)[0]) - U3[:,:40]@np.transpose(U3[:,:40]))@d)
print(r)
# jämför med alla
