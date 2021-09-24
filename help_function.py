# functions in class function -- for debugging 
def get_expected_improvement(x_new): 
        # do more research on different methods and figure out why choose EI
        # Using estimate from Gaussian surrogate instead of actual function for a new trial data point to avoid cost 
    mean_y_new, sigma_y_new = test_gauss_pr.predict(np.array([x_new]), return_std=True)
    sigma_y_new = sigma_y_new.reshape(-1,1)
    if sigma_y_new < 0.0:
        return 0.0
        
        # Using estimates from Gaussian surrogate instead of actual function for entire prior distribution to avoid cost
        # add a very small to avoid divide by zero sigma 
    mean_y = test_gauss_pr.predict(init_X)
    max_mean_y = np.max(mean_y)
    z = (mean_y_new - max_mean_y) / (sigma_y_new+1e-9)
    exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        
    return exp_imp
    
def acquisition_function(x):
        # max (function) = min (-function)
    return -(get_expected_improvement(x))

def random_select_inputs(batch_size):
    theta = np.random.choice(np.arange(120,160),batch_size)
    phi = np.random.choice(np.arange(60,120),batch_size)
    U = np.random.uniform(low = 0.05, high = 0.7, size = batch_size)
    mu = np.random.choice(np.arange(16,60),batch_size)
    g = np.random.choice(np.arange(250,500)*1e-6,batch_size)
    r = np.random.uniform(low = 1.5, high = 4, size = batch_size)
    random_matrix = np.stack([theta, phi, U, mu, g, r], axis=1)
    return random_matrix

def get_next_probable_point_test():
    min_ei = float(sys.maxsize)
    x_optimal = None 
    bounds = [(1e-6, 141), (1e-6, 81), (1e-6, 0.51), (1e-6, 60), (1e-6, 500*1e-6), (1e-6, 4.1)]        
    # Trial with an array of input_simulated_data
    random_matrix = random_select_inputs(30)
    for x_start in random_matrix:
        response = minimize(fun=acquisition_function, x0=x_start, method='L-BFGS-B',bounds = bounds)
        print(x_optimal)
        if response.fun < min_ei:
            min_ei = response.fun
            x_optimal = response.x
            
    return x_optimal, min_ei

def extend_prior_with_posterior_data(x,y):
        
    init_X = np.append(init_X, np.array([x]), axis = 0)
    init_Y = np.append(init_Y, np.array([y]), axis = 0)
    
def optimize():
    distances_ = []
    best_samples = []
    y_max_ind = np.argmax(init_Y)
    y_max = init_Y[y_max_ind]
    optimal_x = init_X[y_max_ind]
    optimal_ei = None
    for i in range(2):
        test_gauss_pr.fit(init_X, init_Y)
        x_next, ei = get_next_probable_point_test()
        y_next = objective(np.array([x_next]))[0]
        extend_prior_with_posterior_data(x_next,y_next)
        if y_next > y_max:
            y_max = y_next
            optimal_x = x_next
            optimal_ei = ei

        if i == 0:
            prev_x = x_next
        else:
            distances_.append(np.linalg.norm(prev_x - x_next))
            prev_x = x_next
            
        best_samples = best_samples.append({"y": y_max, "ei": optimal_ei},ignore_index=True)
        
    return optimal_x, y_max