import numpy as np
import matplotlib.pyplot as plt


# Propensity functions (for predator-prey model)
def PropensityFunc(State,ReactNo):
    # State: state variable [F,R]
    # ReactNo: number of reactions
    w = np.zeros(ReactNo)
    F,R = State
    w[0] = alpha*R
    w[1] = beta*F*R
    w[2] = gamma*F
    return w






def SSA(Initial, StateChangeMat, FinalTime):
   # Inputs:
   #  Initial: initial conditins of size (StateNo x 1)
   #  StateChangeMat: State-change matrix of size (ReactNo, StateNo)
   #  FinalTime: the maximum time we want the process be run

   # Output:
   #  AllTimes: all selected time levels
   #  AllStates: all state values at corresponding time levels

    [m,n] = StateChangeMat.shape
    ReactNum = np.array(range(m))
    AllTimes = {}   # define a dictionary for storing all time levels
    AllStates = {}  # define a dictionary for storing all states at all time levels
    AllStates[0] = Initial
    AllTimes[0] = [0]
    k = 0; t = 0; State = Initial
    while True:
        w = PropensityFunc(State, m)     # propensities
        a = np.sum(w)
        tau = RandExp(a,1)               # WHEN the next reaction happens
        t = t + tau                      # update time
        if t > FinalTime:
            break
        which = RandDisct(ReactNum,w/a,1)             # WHICH reaction occurs
        State = State + StateChangeMat[which.item(),] # Uppdate the state
        k += 1
        AllTimes[k] = t
        AllStates[k] = State
    return AllTimes, AllStates






alpha, beta, gamma = 1, 0.005, 0.6  # constant rates
Initial = [50,100]                  # Initial number of foxes and rabbits
FinalTime = 30                      # final time of simulation
                                    # State-Change Matrix
                        #   A  R  C Da Dr Da´ Dr´ Ma Mr
StateChangeMat = np.array([
                          [-1,  -1, 1, 0, 0, 0, 0, 0 ,0],
                          [-1,  0, 0, 0, 0, 0, 0, 0 ,0],
                          [0,  1, -1, 0, 0, 0, 0, 0 ,0],
                          [0,  -1, 0, 0, 0, 0, 0, 0 ,0],
                          [-1,  0, 0, -1, 0, 1, 0, 0 ,0],
                          [-1,  0, 0, 0, -1, 0, 1, 0 ,0],
                          [1,  0, 0, 1, 0, -1, 0, 0 ,0],
                          [0,  0, 0, 0, 0, 0, 0, 1 ,0],
                          [0,  0, 0, 0, 0, 0, 0, 1 ,0],
                          [0,  0, 0, 0, 0, 0, 0, -1 ,0],
                          [1,  0, 0, 0, 0, 0, 0, 0 ,0]
                          [1,  0, 0, 0, 1, 0, -1, 0 ,0],
                          [0,  0, 0, 0, 0, 0, 0, 0 ,1],
                          [0,  0, 0, 0, 0, 0, 0, 0 ,1],
                          [0,  0, 0, 0, 0, 0, 0, 0 ,-1],
                          [0,  1, 0, 0, 0, 0, 0, 0 ,0], ])