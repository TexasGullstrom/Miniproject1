import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load("TrainDigits.npy")
TrainLab = np.load("TrainLabels.npy")

TestDig = np.load("TestDigits.npy")
TestLab = np.load("TestLabels.npy")




#siffran 0
index = (TrainLab == 0); # find train digits of type 0
A0 = TrainMat[:,index[0]] # all train digits of type 0
A0 = A0[:,0:400] # the first 400 train digits of type 0
U0,S0,Vt0 = np.linalg.svd(A0) # SVD

#siffran 1
index = (TrainLab == 1); # find train digits of type 1
A1 = TrainMat[:,index[0]] # all train digits of type 1
A1 = A1[:,0:400] # the first 400 train digits of type 1
U1,S1,Vt1 = np.linalg.svd(A1) # SVD

#siffran 2
index = (TrainLab == 2); # find train digits of type 2
A2 = TrainMat[:,index[0]] # all train digits of type 2
A2 = A2[:,0:400] # the first 400 train digits of type 2
U2,S2,Vt2 = np.linalg.svd(A2) # SVD

#siffran 3
index = (TrainLab == 3); # find train digits of type 3
A3 = TrainMat[:,index[0]] # all train digits of type 3
A3 = A3[:,0:400] # the first 400 train digits of type 3
U3,S3,Vt3 = np.linalg.svd(A3) # SVD

#siffran 4
index = (TrainLab == 4); # find train digits of type 4
A4 = TrainMat[:,index[0]] # all train digits of type 4
A4 = A4[:,0:400] # the first 400 train digits of type 4
U4,S4,Vt4= np.linalg.svd(A4) # SVD

#siffran 5
index = (TrainLab == 5); # find train digits of type 5
A5 = TrainMat[:,index[0]] # all train digits of type 5
A5 = A5[:,0:400] # the first 400 train digits of type 5
U5,S5,Vt5 = np.linalg.svd(A5) # SVD

#siffran 6
index = (TrainLab == 6); # find train digits of type 6
A6 = TrainMat[:,index[0]] # all train digits of type 6
A6 = A6[:,0:400] # the first 400 train digits of type 6
U6,S6,Vt6 = np.linalg.svd(A6) # SVD

#siffran 2
index = (TrainLab == 7); # find train digits of type 7
A7 = TrainMat[:,index[0]] # all train digits of type 7
A7 = A7[:,0:400] # the first 400 train digits of type 7
U7,S7,Vt7 = np.linalg.svd(A7) # SVD

#siffran 8
index = (TrainLab == 8); # find train digits of type 8
A8 = TrainMat[:,index[0]] # all train digits of type 8
A8 = A8[:,0:400] # the first 400 train digits of type 8
U8,S8,Vt8 = np.linalg.svd(A8) # SVD

#siffran 9
index = (TrainLab == 9); # find train digits of type 9
A9 = TrainMat[:,index[0]] # all train digits of type 9
A9 = A9[:,0:400] # the first 400 train digits of type 9
U9,S9,Vt9 = np.linalg.svd(A9) # SVD


if True:
    plt.plot(np.linspace(0,len(S0),len(S0)),S0) #Singular values for 0
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S1) #Singular values for 1
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S2) #Singular values for 2
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S3) #Singular values for 3
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S4) #Singular values for 4
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S5) #Singular values for 5
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S6) #Singular values for 6
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S7) #Singular values for 7
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S8) #Singular values for 8
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S9) #Singular values for 9
    plt.show()


if False: # Visar den perfekta 3:an
    A1= np.dot(U3[:,:1] * S3[:1], Vt3[:1,:])
    D3 = np.reshape(A1[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D3, cmap ="gray") # Den perfekta 3:an
    plt.show()
 
d = TestDig[:,7] # en testbild

if False:
    D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D, cmap ="gray") # Plot of the digit
    plt.show()


#x = np.linalg.solve(U,A3)

# Beräkna residual



# beräkna residual, exempel 
k=15
r = np.linalg.norm((np.eye(np.shape(U3)[0]) - U3[:,:k]@np.transpose(U3[:,:k]))@d)
print(r)
# jämför med alla

