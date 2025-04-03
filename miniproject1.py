from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as plt


def genetic_oscillators():
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

    yprime=np.zeros(9)
    yprime[0] = theta_A
    yprime[1] = 
    return yprime

teval = np.linspace(0, FinalTime,1000)      # fine evaluation time samples
sol = solve_ivp(Predator_Prey, [0,FinalTime], Initial, method = 'BDF', t_eval = teval)

plt.figure(figsize = (6, 3))
plt.plot(sol.t,sol.y[0],linestyle = 'solid', color='blue', label = 'Rabbits')
plt.plot(sol.t,sol.y[1],linestyle = 'solid', color='red', label = 'Foxes')
plt.xlabel('Time'); plt.ylabel('Number of animals')
plt.title('Deterministic solution using BDF')
plt.legend(loc='upper right')
plt.show()
