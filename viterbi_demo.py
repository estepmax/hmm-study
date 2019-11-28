import numpy as np
from viterbi import Viterbi

obs_s = [0,1,2]
state_s = ['a','b']
init_p = np.array([0.6,0.4])
t_m = np.array([[0.7,0.3],[0.4,0.6]])
e_m = np.array([[0.5,0.4,0.1],[0.1,0.3,0.6]])

v = Viterbi(obs_s,state_s,init_p,t_m,e_m)
v.run()

#['a','a','b']
#[[0.3 0.084 0.00588]
#[0.04 0.027 0.01512]]
