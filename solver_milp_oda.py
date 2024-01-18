from scipy.optimize import minimize
import numpy as np


num_gpu = 8

models = ['0', '5', '10', '15', '20', '25']
candidate_machine_max_throughput = [1.65*14.29, 1.65*15.87, 1.65*17.86, 1.65*20.41, 1.65*23.81, 1.65*28.57]  # - query_per_minute
candidate_machine_acc = [100.0, 97.046, 95.808, 94.378, 93.235, 91.52]

def solver(total_load):
    while total_load in []:
        print(total_load)
        total_load = total_load+1
        print(total_load)
        
    # Load constraint
    total_load = total_load  # Adjust as needed
    
    
    load_constraints = ({'type': 'eq', 'fun': lambda y: total_load - sum(y)})
    
    # Throughput constraints
    throughput_constraints = [{'type': 'ineq', 'fun': lambda x_y, i=i: candidate_machine_max_throughput[int(x_y[:num_gpu][i])] - x_y[num_gpu:][i]} for i in range(num_gpu)]
    
    # Objective function to maximize overall accuracy
    def objective_function(x_y):
        x = [int(round(val)) for val in x_y[:num_gpu]]  # Rounded to the nearest integer
        y = x_y[num_gpu:]    
        current_load = total_load - sum(y)

        # Check each throughput constraint
        for i in range(num_gpu):
            throughput_used = x_y[num_gpu:][i]
            max_throughput = candidate_machine_max_throughput[int(x[i])]

        obj_value = -sum(y[i] * candidate_machine_acc[x[i]] for i in range(num_gpu))
        return obj_value
    
    # Initial guess
    initial_guess = [i for i in range(num_gpu)] + [total_load / num_gpu] * num_gpu
    
    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, constraints=[load_constraints] + throughput_constraints, bounds=[(0, len(models) - 1)] * num_gpu + [(0, None)] * num_gpu,  tol=1e-6)
    
    # Extracting the solution
    solution_x = [int(round(val)) for val in result.x[:num_gpu]]  # Rounded to the nearest integer
    solution_y = result.x[num_gpu:]
    return solution_x, solution_y, sum(solution_y), -result.fun


def oda_algorithm(hk, fk):
    Probabiltiy = {}

    for i in range(0,6):
        for j in range(0,6):
            Probabiltiy[(i,j)] = 0
            if i == j:
                Probabiltiy[(i,j)] = 1
 
    h_old = np.copy(hk)
    for ki in range(5, 0, -1):
        if hk[ki] > fk[ki]:
            P = (hk[ki] - fk[ki]) / hk[ki]
            sum_others = 0
            for next_one in range(ki+1, 6):
                if Probabiltiy[(ki,next_one)]!=0:
                    sum_others+=Probabiltiy[(ki,next_one)]
  
            Probabiltiy[(ki,ki-1)] = P * (1-sum_others)
            Probabiltiy[(ki,ki)] = (1 - P) * (1-sum_others)
            hk[ki-1] += hk[ki] - fk[ki]
            hk[ki] = fk[ki]
        else:
            j = 1
            while round(hk[ki],2) < round(fk[ki],2):
                shift = min(hk[ki-j], fk[ki] - hk[ki])
                P = shift / h_old[ki-j]

                hk[ki-j] -= shift
                hk[ki] += shift

                Probabiltiy[(ki-j,ki)] = P
                Probabiltiy[(ki-j,ki-j)] = hk[ki-j]/h_old[ki-j]
                
                j += 1
    return Probabiltiy
