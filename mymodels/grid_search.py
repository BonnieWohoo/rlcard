# In grid_search.py
import itertools

from testing_dql import train  # Import your train function

# Define the fixed parameters (other than the ones being tested)
fixed_params = {
    "replay_memory_size": 20000,
    "replay_memory_init_size": 100,
    "update_target_estimator_every": 1000,
    "train_every": 1,
    "num_actions": 2,
    "state_shape": None,
    "device": None,
    "save_path": None,
    "save_every": float('inf'),
}

# Define the parameter values to test
epsilon_start_values = [0.1, 0.2, 0.5, 0.7, 1.0]
epsilon_end_values = [0.05, 0.1, 0.2, 0.3]
batch_size_values = [32, 64, 128]
learning_rate_values = [0.00001, 0.00005, 0.0001, 0.001, 0.01]
discount_factor_values = [0.90, 0.95, 0.98, 0.99, 1.0]
mlp_layers_values = [[64, 64], [128, 128], [64, 64, 64], [128, 64, 32]]
epsilon_decay_steps_values = [10000, 20000, 50000]


# Grid search function
def test_all_param_combinations():
    param_combinations = itertools.product(
        epsilon_start_values,
        epsilon_end_values,
        batch_size_values,
        learning_rate_values,
        discount_factor_values,
        mlp_layers_values,
        epsilon_decay_steps_values
    )
    
    results = []
    
    for (epsilon_start, epsilon_end, batch_size, learning_rate, 
         discount_factor, mlp_layers, epsilon_decay_steps) in param_combinations:
        
        print(f"Testing with parameters:")
        print(f"epsilon_start={epsilon_start}, epsilon_end={epsilon_end}, batch_size={batch_size}, "
              f"learning_rate={learning_rate}, discount_factor={discount_factor}, "
              f"mlp_layers={mlp_layers}, epsilon_decay_steps={epsilon_decay_steps}")
        
        # Update the parameters with the current combination
        params = fixed_params.copy()
        params["epsilon_start"] = epsilon_start
        params["epsilon_end"] = epsilon_end
        params["batch_size"] = batch_size
        params["learning_rate"] = learning_rate
        params["discount_factor"] = discount_factor
        params["mlp_layers"] = mlp_layers
        params["epsilon_decay_steps"] = epsilon_decay_steps
        
        # Run the training with the current parameter set
        result = train(params, run_id=f"epsilon_{epsilon_start}_end_{epsilon_end}_batch_{batch_size}_lr_{learning_rate}_"
                                     f"discount_{discount_factor}_layers_{mlp_layers}_decay_{epsilon_decay_steps}")
        results.append((epsilon_start, epsilon_end, batch_size, learning_rate, discount_factor, mlp_layers, epsilon_decay_steps, result))  # Store the result
        
    return results


# Call the grid search function
all_results = test_all_param_combinations()

# Print all results
for result in all_results:
    print(result)
