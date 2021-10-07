import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import math
from scipy.stats import norm
from scipy.optimize import minimize 
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
# read data file
nn_df = pd.read_csv("Non_Newtonian_Simulation_V1.csv").iloc[:,1:]
input_list = ["theta_angle","phi_angle","Web_speed(m/s)","ink_viscosity(cP)","coating_height(um)","ratio"]
X_array_nn = nn_df[[*input_list]]
X_array_nn["coating_height(um)"] = X_array_nn["coating_height(um)"].apply(lambda x:x*1e-6)

property_list = [0.53,0.25,0.065,1143.5]
x_f = 0.0005
x_d = 0.101
h_w = 0.1
quantile = 0.5

def objective(x_input,ink_value=property_list,judge = x_f, qt = quantile,Xd =x_d):
    """
    objective function to max rewards 
    
    design this function with quantile weight, inspired by quantle loss regression idea and rewards matrix used in the RL notebook

    for slot die coating, two edge cases are: leaking, right: breaking-up

    leakings are zero-tolerance defects. goal is to encourage more prediction points on right side

    R(x_u,x_f) = (x_f - x_u)*tao if x_u - half point > 0;

               = (x_u - 0)*(1-tao) if x_u - half point < 0 
    
    interpretation : if tao = 0.1, then R will be larger for points located on left side, which potentially increase chance to make x_u on left part of middle point
    goal is to max rewards 
   
    """
    # get output from input, input is a matrix with 6 columns
    # get ink properties:
    n,m,sigma,density = ink_value
    y = []
#     est_x = []
    for x_i in x_input: 
    # searching values and doing calculations
        
        theta, phi, U, mu, g, r = x_i
        thickness = g/r  
        fr = thickness*(U*h_w)
            
        a = (U*(n+1)*(2*n+1)/n)**n*(g**(-n-1))
        b_s = (U*(g-2*thickness)*(n+1)*(2*n+1)/n)**n * (g**(-2*n-1))
        b_l = -(U*(2*thickness-g)*(n+1)*(2*n+1)/n)**n * (g**(-2*n-1))
        modified_CA = ((U*m*(U/g)**(n-1))/sigma)**(2/3)      
        p_ambient = density*U**2/2

        if thickness <= g/2:
            xu_estimate = judge - (p_ambient - 1.34*modified_CA*sigma/thickness - (Xd-judge)*m*b_s - sigma*(math.cos(theta)+math.cos(phi)))/a*m
#             est_x.append(xu_estimate)
                
        elif thickness > g/2:
            xu_estimate = judge - (p_ambient - 1.34*modified_CA*sigma/thickness - (Xd-judge)*m*b_l - sigma*(math.cos(theta)+math.cos(phi)))/a*m
#             est_x.append(xu_estimate)
        
        if xu_estimate <= 0:
            y.append(xu_estimate*2)
        elif xu_estimate >= judge/2 and xu_estimate < judge:
            y.append((judge - xu_estimate)*qt)
        elif xu_estimate < judge/2 and xu_estimate > 0:
            y.append(xu_estimate*(1-qt))
        else:
            y.append(-xu_estimate*2)
    
    return  y


class Bayesian_opt():
    def __init__(self, target_func, x_init, y_init, n_iter, batch_size):
        # initial values
        self.x_init = x_init
        self.y_init = y_init
        self.n_iter = n_iter
        self.target_func = target_func
        self.batch_size = batch_size  
        
        self.best_samples = pd.DataFrame(columns = (input_list + ["y","ei"]))
        self.gauss_pr = GaussianProcessRegressor()
        self.distances_ = []
    # surrogate function using EI method, as a starting point  
    def _get_expected_improvement(self, x_new): 
        # do more research on different methods and figure out why choose EI
        # Using estimate from Gaussian surrogate instead of actual function for a new trial data point to avoid cost 
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1,1)
        if sigma_y_new < 0.0:
            return 0.0
        
        # Using estimates from Gaussian surrogate instead of actual function for entire prior distribution to avoid cost
        # add a very small to avoid divide by zero sigma 
        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / (sigma_y_new+1e-9)
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        
        return exp_imp
    
    def acquisition_function(self,x):
        # max (function) = min (-function)
        return -self._get_expected_improvement(x)
    
    def random_select_inputs(self):
        theta = np.random.choice(np.arange(120,140),self.batch_size)
        phi = np.random.choice(np.arange(60,80),self.batch_size)
        U = np.random.uniform(low = 0.05, high = 0.5, size = self.batch_size)
        mu = np.random.choice(np.arange(16,60),self.batch_size)
        g = np.random.choice(np.arange(250,500)*1e-6,self.batch_size)
        r = np.random.uniform(low = 1.5, high = 4, size = self.batch_size)
        random_matrix = np.stack([theta, phi, U, mu, g, r], axis=1)
        return random_matrix

    def _get_next_probable_point(self):
            min_ei = float(sys.maxsize)
            x_optimal = None 
            bounds = [(1e-6, 141), (1e-6, 81), (1e-6, 0.51), (1e-6, 60), (1e-6, 500*1e-6), (1e-6, 4.1)]
            # Trial with an array of input_simulated_data
            random_matrix = self.random_select_inputs()
            for x_start in random_matrix:
                response = minimize(fun=self.acquisition_function, x0=x_start, method='L-BFGS-B',bounds = bounds)
                if response.fun < min_ei:
                    min_ei = response.fun
                    x_optimal = response.x
            
            return x_optimal, min_ei  
        
    # wait for debug for rest of two functions     
    def _extend_prior_with_posterior_data(self, x,y):
        
        self.x_init = np.append(self.x_init, np.array([x]), axis = 0)
        self.y_init = np.append(self.y_init, np.array([y]), axis = 0)
  
    def optimize(self):
        y_max_ind = np.argmax(self.y_init)
        y_max = self.y_init[y_max_ind]
        optimal_x = self.x_init[y_max_ind]
        optimal_ei = None
        for i in range(self.n_iter):
            self.gauss_pr.fit(self.x_init, self.y_init)
            x_next, ei = self._get_next_probable_point()
            y_next = self.target_func(np.array([x_next]))[0]
            self._extend_prior_with_posterior_data(x_next,y_next)
            # 
            if y_next > y_max:
                y_max = y_next
                optimal_x = x_next
                optimal_ei = ei

            if i == 0:
                 prev_x = x_next
            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next
            
            self.best_samples = self.best_samples.append({"theta_angle":optimal_x[0],"phi_angle":optimal_x[1],"Web_speed(m/s)":optimal_x[2],
                                                           "ink_viscosity(cP)":optimal_x[3],"coating_height(um)":optimal_x[4],
                                                           "ratio":optimal_x[5],"y": y_max, "ei": optimal_ei},ignore_index=True)
        
        return optimal_x, y_max
    
# start to train
init_X = X_array_nn_samples.values
init_Y = objective(X_array_nn_samples.values)

bopt = Bayesian_opt(target_func=objective, x_init=init_X, y_init=init_Y, n_iter=100, batch_size=30)
bopt.optimize()

bopt.best_samples.to_csv("best_samples_bayesian.csv")
