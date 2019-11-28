import numpy as np

class Viterbi(object):
    
    '''
    args:
        obs_s: observation space
        state_s: state space
        init_p: initial probabilities
        t_m: transition matrix
        e_m: emission matrix
    '''

    def __init__(self,obs_s,state_s,init_p,t_m,e_m):
        self.obs_s = obs_s
        self.state_s = state_s
        self.init_p = init_p
        self.t_m = t_m
        self.e_m = e_m
    
    '''
    def argmax2d(self,m): #used to retrieve the 2D analogue of 1D argmax
        idx = np.unravel_index(np.argmax(m,axis=None),m.shape)
        return idx
    '''
 
    def run(self):
       
        s_n = len(self.state_s) #size of state space
        o_n = len(self.obs_s) #size of observation space
        
        values = np.zeros((s_n,o_n)) #table of values  
        indicies = np.empty((s_n,o_n),object) #table of indicies 
        V = np.stack((values,indicies),axis=2) #create 3D array with two dtypes for values and indicies
        
        for i in range(s_n):  
            V[i,0,0] = self.init_p[i]*self.e_m[i,0]
        
        for j in range(1,o_n):
            for i in range(s_n):
                V[i,j,0] = np.max(V[:,j-1,0]*self.t_m[:,i]*self.e_m[i,j]) 
                V[i,j,1] = np.argmax(V[:,j-1,0]*self.t_m[:,i])
         
        z_t = np.argmax(V[:,o_n-1,0]) 
        x_t = [self.state_s[z_t]] 

        for i in reversed(range(1,o_n)): #backtrack through states
            z_t = V[z_t,i,1]
            x_t.append(self.state_s[z_t])
       
        print(x_t[::-1])
        print(V[:,:,0])


