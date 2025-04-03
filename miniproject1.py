from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

FinalTime=400

theta_A=50
theta_R=100
Dprime_A=0
D_A=1
Dprime_R=0
D_R=1
M_A=0
C=0
R=0
M_R=0
A=0
gamma_A=1
gamma_R=1
gamma_C=2
alphaprime_A=500
alpha_A=50
alpha_R=0.01
alphaprime_R=50
delta_MA=10
dela_R=0.2
delta_MR=0.5
delta_A=1
beta_A=50
beta_R=5

Initial= [D_A,D_R,Dprime_A,Dprime_R,M_A,A,M_R,R,C]

def yprime(t,y):
    D_A,D_R,Dprime_A,Dprime_R,M_A,A,M_R,R,C = y
    yprime=np.zeros(9)
    yprime[0] = theta_A*Dprime_A-gamma_A*D_A*A
    yprime[1] = theta_R*Dprime_R-gamma_R*D_R*A
    yprime[2] = gamma_A*D_A*A-theta_A*Dprime_A
    yprime[3] = gamma_R*D_R*A-theta_R*Dprime_A
    yprime[4] = alphaprime_A*D_A + alpha_A*D_A - delta_MA*M_A
    yprime[5] = beta_A*M_A + theta_A*Dprime_A + theta_R*Dprime_R - A*(gamma_A*D_A + gamma_R*D_R + gamma_C*R + delta_A)
    yprime[6] = alphaprime_A*D_R + alpha_R*D_R - delta_MR*M_R
    yprime[7] = beta_R*M_R - gamma_C*A*R + delta_A*C - dela_R*R
    yprime[8] = gamma_C*A*R - delta_A*C 
    return yprime


teval = np.linspace(0, FinalTime,50)      # fine evaluation time samples
sol = solve_ivp(yprime, [0,FinalTime], Initial, method = 'BDF', t_eval = teval)

plt.figure(figsize = (6, 3))
plt.plot(sol.t,sol.y[5],linestyle = 'solid', color='blue', label = 'Rabbits')
plt.plot(sol.t,sol.y[7],linestyle = 'solid', color='red', label = 'Foxes')
plt.xlabel('Time'); plt.ylabel('Number of animals')
plt.title('Deterministic solution using BDF')
plt.legend(loc='upper right')

# Save the figure to a file
plt.show()



