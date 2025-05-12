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



# Ritar singular value för 3 och 8
if False:

    plt.plot(np.linspace(0,len(S0),len(S0)),S3) #Singular values for 3
    plt.xlabel("k-index")
    plt.xlabel("Singular value")
    plt.title("Singular value for the digit 3")
    plt.show()



    plt.plot(np.linspace(0,len(S0),len(S0)),S8) #Singular values for 8
    plt.xlabel("k-index")
    plt.xlabel("Singular value")
    plt.title("Singular value for the digit 8")
    plt.show()


if False: # Visar den perfekta 3:an
    # siffran 3 
    # u1
    A3_1= np.dot(U3[:,:1] * S3[:1], Vt3[:1,:])
    D3_1 = np.reshape(A3_1[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,1)
    plt.imshow(D3_1, cmap ="gray") # Den perfekta 3:an
    plt.title("Singular image u1 of the digit 3")

    
    # u2 
    A3_2= np.dot(U3[:,:2] * S3[:2], Vt3[:2,:])
    D3_2 = np.reshape(A3_2[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,2)
    plt.imshow(D3_2, cmap ="gray") # Den perfekta 3:an
    plt.title("Singular image u2 of the digit 3")


    #u3
    A3_3= np.dot(U3[:,:3] * S3[:3], Vt3[:3,:])
    D3_3 = np.reshape(A3_3[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,3)
    plt.imshow(D3_3, cmap ="gray") # Den perfekta 3:an
    plt.title("Singular image u3 of the digit 3")
    plt.show()


if True: # Visar den perfekta 8:an
    # siffran 3 
    # u1
    A8_1= np.dot(U8[:,:1] * S8[:1], Vt8[:1,:])
    D8_1 = np.reshape(A8_1[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,1)
    plt.imshow(D8_1, cmap ="gray") # Den perfekta 8:an
    plt.title("Singular image u1 of the digit 8")

    
    # u2 
    A8_2= np.dot(U8[:,:2] * S8[:2], Vt8[:2,:])
    D8_2 = np.reshape(A8_2[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,2)
    plt.imshow(D8_2, cmap ="gray") # Den perfekta 8:an
    plt.title("Singular image u2 of the digit 8")


    #u3
    A8_3= np.dot(U8[:,:3] * S8[:3], Vt8[:3,:])
    D8_3 = np.reshape(A8_3[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.subplot(1,3,3)
    plt.imshow(D8_3, cmap ="gray") # Den perfekta 8:an
    plt.title("Singular image u3 of the digit 8")
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