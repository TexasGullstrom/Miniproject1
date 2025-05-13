import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load("TrainDigits.npy")
TrainLab = np.load("TrainLabels.npy")

TestDig = np.load("TestDigits.npy")
TestLab = np.load("TestLabels.npy")


if True: #SVD,Test Digits
#siffran 0
    index = (TrainLab == 0) # find train digits of type 0
    A0 = TrainMat[:,index[0]] # all train digits of type 0
    A0 = A0[:,0:400] # the first 400 train digits of type 0
    U0,S0,Vt0 = np.linalg.svd(A0) # SVD
    index = (TestLab == 0) # find train digits of type 0
    testdigits_0= TestDig[:,index[0]] #all test digits of type 0


    #siffran 1
    index = (TrainLab == 1)
    A1 = TrainMat[:,index[0]] 
    A1 = A1[:,0:400]
    U1,S1,Vt1 = np.linalg.svd(A1)
    index= (TestLab == 1) 
    testdigits_1= TestDig[:,index[0]]


    #siffran 2
    index = (TrainLab == 2); 
    A2 = TrainMat[:,index[0]] 
    A2 = A2[:,0:400]
    U2,S2,Vt2 = np.linalg.svd(A2) 
    index= (TestLab == 2) 
    testdigits_2= TestDig[:,index[0]]


    #siffran 3
    index = (TrainLab == 3); 
    A3 = TrainMat[:,index[0]] 
    A3 = A3[:,0:400] 
    U3,S3,Vt3 = np.linalg.svd(A3)
    index= (TestLab == 3) 
    testdigits_3= TestDig[:,index[0]]
 

    #siffran 4
    index = (TrainLab == 4); 
    A4 = TrainMat[:,index[0]] 
    A4 = A4[:,0:400]
    U4,S4,Vt4= np.linalg.svd(A4)
    index= (TestLab == 4) 
    testdigits_4= TestDig[:,index[0]]

    #siffran 5
    index = (TrainLab == 5); 
    A5 = TrainMat[:,index[0]] 
    A5 = A5[:,0:400] 
    U5,S5,Vt5 = np.linalg.svd(A5) 
    index= (TestLab == 5) 
    testdigits_5= TestDig[:,index[0]]


    #siffran 6
    index = (TrainLab == 6) 
    A6 = TrainMat[:,index[0]]
    A6 = A6[:,0:400] 
    U6,S6,Vt6 = np.linalg.svd(A6) 
    index= (TestLab == 6) 
    testdigits_6= TestDig[:,index[0]]


    #siffran 7
    index = (TrainLab == 7)
    A7 = TrainMat[:,index[0]]
    A7 = A7[:,0:400]
    U7,S7,Vt7 = np.linalg.svd(A7)
    index= (TestLab == 7) 
    testdigits_7= TestDig[:,index[0]]


    #siffran 8
    index = (TrainLab == 8)
    A8 = TrainMat[:,index[0]] 
    A8 = A8[:,0:400]
    U8,S8,Vt8 = np.linalg.svd(A8) 
    index= (TestLab == 8) 
    testdigits_8= TestDig[:,index[0]]
   

    #siffran 9
    index = (TrainLab == 9)
    A9 = TrainMat[:,index[0]] 
    A9 = A9[:,0:400] 
    U9,S9,Vt9 = np.linalg.svd(A9) 
    index= (TestLab == 9) 
    testdigits_9= TestDig[:,index[0]]

U=np.concatenate((U0,U1,U2,U3,U4,U5,U6,U7,U8,U9),axis=1)
k=np.linspace(1,40,40)
TestSolved=np.zeros((40000,1))
isequal=np.zeros((0,40))

if False: #Fungerande kod
    for j in range(0,100):
        r=10000
        for i in range(0,10):
            r_i=np.linalg.norm((np.eye(784) - U[:,784*i:784*(i+1)-1][:,:k]@np.transpose(U[:,784*i:784*(i+1)-1][:,:k]))@TestDig[:,j])
            if r_i<r:
                r=r_i
                Digit=i
            if i==9:#Return the assumed number
                TestSolved[j]=Digit 


    isequal= np.zeros((100,1))
    for i in range(0,len(TestSolved)):
        if TestSolved[i] == TestLab[:,i]:
            isequal[i]=1

    print(np.sum(isequal))


for p in range(0,40):
    r=np.linalg.norm((np.eye(784) - U0[:,:int(k[p])]@np.transpose(U0[:,:int(k[p])]))@TestDig,axis=0)

    for i in range(0,9):
        r_i=np.linalg.norm((np.eye(784) - U[:,784*(i+1):784*(i+2)-1][:,:int(k[p])]@np.transpose(U[:,784*(i+1):784*(i+2)-1][:,:int(k[p])]))@TestDig,axis=0)
        r=np.vstack((r,r_i))

    TestSolved=np.argmin(r,axis=0)
    isequal_i= np.zeros((40000,1))
    for i in range(0,len(TestSolved)):
        if TestSolved[i] == TestLab[:,i]:
            isequal_i[i]=1
    isequal[p]=np.sum(isequal_i)/40000

plt.plot(k,isequal)
plt.show()

if False: #Plot Singlular values
    plt.plot(np.linspace(0,len(S0),len(S0)),S0) #Singular values for 0
    plt.show()
    plt.plot(np.linspace(0,len(S0),len(S0)),S1) #Singular values for 1
    plt.show()
   


if False: # Visar den perfekta 3:an
    A1= np.dot(U3[:,:1] * S3[:1], Vt3[:1,:])
    D3 = np.reshape(A1[:,1], (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D3, cmap ="gray") # Den perfekta 3:an
    plt.show()
 

if False:
    D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D, cmap ="gray") # Plot of the digit
    plt.show()


#x = np.linalg.solve(U,A3)

# Beräkna residual


# beräkna residual, exempel 

# jämför med alla


