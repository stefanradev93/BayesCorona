import numpy as np

rh_names = [r'rho1',r'rho2',r'rho3',r't1',r't2',r't3',r't4']
th_names = [r'E0', r'p', r'$\beta$', r'$\alpha$', r'$\gamma$', r'$d$']

th_low = [1,0.01, 0.8, 0.05, 0.05, 0.0001]
th_high=[100, 1.0, 3.25, 0.7, 0.3, 0.05]
rh_low = [0,0,0,0,0,0,0]
rh_high = [1,1,1,1,1,1,1]

def prior_sample(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    
    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------
    
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """
    

    
    theta = np.random.uniform(th_low, 
                              th_high, size=(batch_size, len(th_low)))
    
    return theta