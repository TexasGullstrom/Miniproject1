import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load("TrainDigits.npy")
TrainLab = np.load("TrainLabels.npy")

TestDig = np.load("TestDigits.npy")
TestLab = np.load("TestLabels.npy")


if True: #SVD
#siffran 0
    index = (TrainLab == 0) # find train digits of type 0
    A0 = TrainMat[:,index[0]] # all train digits of type 0
    A0 = A0[:,0:400] # the first 400 train digits of type 0
    U0,S0,Vt0 = np.linalg.svd(A0) # SVD


    #siffran 1
    index = (TrainLab == 1)
    A1 = TrainMat[:,index[0]] 
    A1 = A1[:,0:400]
    U1,S1,Vt1 = np.linalg.svd(A1)


    #siffran 2
    index = (TrainLab == 2); 
    A2 = TrainMat[:,index[0]] 
    A2 = A2[:,0:400]
    U2,S2,Vt2 = np.linalg.svd(A2) 


    #siffran 3
    index = (TrainLab == 3); 
    A3 = TrainMat[:,index[0]] 
    A3 = A3[:,0:400] 
    U3,S3,Vt3 = np.linalg.svd(A3)
 

    #siffran 4
    index = (TrainLab == 4); 
    A4 = TrainMat[:,index[0]] 
    A4 = A4[:,0:400]
    U4,S4,Vt4= np.linalg.svd(A4)


    #siffran 5
    index = (TrainLab == 5); 
    A5 = TrainMat[:,index[0]] 
    A5 = A5[:,0:400] 
    U5,S5,Vt5 = np.linalg.svd(A5) 


    #siffran 6
    index = (TrainLab == 6) 
    A6 = TrainMat[:,index[0]]
    A6 = A6[:,0:400] 
    U6,S6,Vt6 = np.linalg.svd(A6) 


    #siffran 7
    index = (TrainLab == 7)
    A7 = TrainMat[:,index[0]]
    A7 = A7[:,0:400]
    U7,S7,Vt7 = np.linalg.svd(A7)


    #siffran 8
    index = (TrainLab == 8)
    A8 = TrainMat[:,index[0]] 
    A8 = A8[:,0:400]
    U8,S8,Vt8 = np.linalg.svd(A8) 
   

    #siffran 9
    index = (TrainLab == 9)
    A9 = TrainMat[:,index[0]] 
    A9 = A9[:,0:400] 
    U9,S9,Vt9 = np.linalg.svd(A9) 


U=np.concatenate((U0,U1,U2,U3,U4,U5,U6,U7,U8,U9),axis=1) 
isequal=np.zeros((10,11)) #Percentage of correctly guessed digits for each k (coloumns) and digits(rows)

for i in range(5,16):
    k=i #Singular values used
    TestSolved=np.zeros((40000,1))
    r=np.linalg.norm((np.eye(784) - U0[:,:int(k)]@np.transpose(U0[:,:int(k )]))@TestDig,axis=0) #first iteration of residuals for test digits 0 to set up a matrix
    #Matrix r is a residual matrix which looks at the residual of each test image compared to each digit. 

    for j in range(0,9):
        r_i=np.linalg.norm((np.eye(784) - U[:,784*(j+1):784*(j+2)-1][:,:int(k)]@np.transpose(U[:,784*(j+1):784*(j+2)-1][:,:int(k )]))@TestDig,axis=0) #Residuals for rest digits
        r=np.vstack((r,r_i)) #Stack unto matrix r

    TestSolved=np.argmin(r,axis=0) #choose digit with smallest residual
    digit_matrix=np.zeros((10,40000)) #Correctly guessed 

    for p in range(0,len(TestSolved)): #Confirm digit choice in each test
        if TestSolved[p] == TestLab[:,p]:
            digit_matrix[TestSolved[p],p]=1

    for m in range(0,10):
        isequal[m,k-5]=np.sum(digit_matrix[m,:])/4000 


plt.figure ( figsize =(10 ,6) )
x_vals = list ( range (5 , 16) )
for i in range (10) :
    plt . plot ( x_vals , isequal [ i,: ] , label = f" Siffra { i }" )
plt . xlabel ( "k - varde ( index fran 0 till 10) " )
plt . ylabel ( " Procent ratt " )
plt . title ( " Procent ratt gissade per siffra over olika k ")
plt . legend ()
plt . grid ( True )
plt . tight_layout ()
plt . show ()

