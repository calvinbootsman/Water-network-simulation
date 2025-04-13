#%%
import wntr
import numpy as np
import threading
import copy
import multiprocessing
import os
import itertools
import csv
inp_file = 'tabletopmodel.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

# set the diameter of the pipes to 26 mm
diameter = 26  # mm
for pipe in wn.pipe_name_list:
    pipe = wn.get_link(pipe)
    pipe.diameter = diameter  # mm
#%%
# these corresponds with where the flow and pressure sensors are located
measurable_nodes = ['22', '15', '4', '3', '8']  
measureble_links = ['35', '33', '5', '18', '9']

# num_of_values = 10
# valve_19_values = np.linspace(0.79, 1.54, num_of_values)
# valve_22_values = np.linspace(0.46, 1.52, num_of_values)
# valve_16_values = np.linspace(0.35, 1.50, num_of_values)
# valve_27_values = np.linspace(0.74, 1.54, num_of_values)
# valve_31_values = np.linspace(0.78, 1.56, num_of_values)

valve_ids_to_modify = ['19', '22', '16', '27', '31']
valve_setting_options = [0.01, 0.2, 1.0, 8, 50, 200, 'closed'] # Valve open: 100%, 80%, 60%, 40%, 20%, 10%, 0%

# Determine number of processes (leave one core free if possible)
try:
    num_cores = os.cpu_count()
    num_processes = max(1, num_cores - 1) if num_cores else 1
except NotImplementedError:
    num_processes = 1 # Default to 1 if cpu_count() is not implemented
print(f"Using {num_processes} worker processes.")
num_processes = 1 # For debugging, set to 1 to avoid multiprocessing issues


# --- Worker Function ---
# This function will be executed by each process in the pool.
# It takes a single combination of settings as input.
def run_simulation_for_combination(settings_combination):
    """
    Creates a copy of the network, applies valve settings, runs simulation.

    Args:
        settings_combination (tuple): A tuple of settings corresponding
                                      to valve_ids_to_modify.

    Returns:
        tuple: A tuple containing (settings_combination, result) or
               (settings_combination, None) if simulation failed.
    """
    try:
        # 1. Create a deep copy *within* the worker process
        # This is crucial for isolation between parallel runs.
        wn_local_copy = copy.deepcopy(wn)

        # 2. Apply the settings for this specific combination
        for valve_id, setting in zip(valve_ids_to_modify, settings_combination):
            link = wn_local_copy.get_link(valve_id)
            if setting == 'closed':
                link.initial_status = wntr.network.LinkStatus.Closed
                # Optional: You might want to set a numeric setting too, e.g., 0
                # link.initial_setting = 0
            else:
                # Ensure the link is treated as open if not 'closed'
                link.initial_status = wntr.network.LinkStatus.Open # Or relevant Open status
                link.initial_setting = setting

        # 3. Run the simulation
        # Ensure 'hours' is accessible here (e.g., global or passed differently)
        result = run_simulation(wn_local_copy, hours)
        # 4. Return the input settings along with the result for tracking
        return (settings_combination, result)

    except Exception as e:
        # Handle potential errors during simulation gracefully
        print(f"Error processing combination {settings_combination}: {e}")
        return (settings_combination, None) # Indicate failure

# 1. Generate all combinations using itertools.product
def main():
    all_combinations = list(itertools.product(valve_setting_options, repeat=len(valve_ids_to_modify)))
    total_simulations = len(all_combinations)
    print(f"Generated {total_simulations} simulation combinations.")

    
    # for i, settings in enumerate(all_combinations):
    #     print(f"Running simulation {i+1}/{total_simulations} with settings: {settings}")
    #     # 1. Run the simulation for this combination
    #     # Note: This is a blocking call, it will wait for the simulation to finish
    #     result = run_simulation_for_combination(settings)
    #     results.append(result)

    # 2. Create the pool and map the worker function to the combinations
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map applies 'run_simulation_for_combination' to each item
        # in 'all_combinations' and distributes the work across processes.
        # It blocks until all results are back.
        results = []
        for i, result in enumerate(pool.imap_unordered(run_simulation_for_combination, all_combinations), start=1):
            if i%100 == 0:
                print(f"Processed combination {i}/{total_simulations}")
            results.append(result)

    print("\n--- Simulation Results ---")
    # 3. Process the collected results
    successful_count = 0
    failed_count = 0

    for i, (settings, flow_result) in enumerate(results):
        if flow_result is not None:
            successful_count += 1
            print(f"Combination {i+1}/{total_simulations}: Settings={settings} -> Result={flow_result}")
            # Print or store the successful result as needed
            # Example: print(f"Combination {i+1}/{total_simulations}: Settings={settings} -> Result={flow_result}")
            pass # Replace 'pass' with your actual result handling
        else:
            failed_count += 1
            print(f"Combination {i+1}/{total_simulations}: Settings={settings} -> FAILED")

    print("\n--- Summary ---")
    print(f"Total simulations attempted: {total_simulations}")
    print(f"Successful simulations: {successful_count}")
    print(f"Failed simulations: {failed_count}")
    return results
    output_file = "simulation_results.npy"
    np.save(output_file, results)
    print(f"Results saved to {output_file}")

#%%
results = []

# if __name__ == "__main__":
results = main()

#%%
print(len(results[0][0]))
print(len(results[0][1]))
print(len(results[0][1][0]))
print(len(results[0][1][1]))
# %%
for i in range(len(results)):
    if len(results[i][0]) != 5:
        print(f'error at i = {i}')
    elif len(results[i][1]) != 2:
        print(f'error at i = {i}')
    elif len(results[i][1][0]) != 5:
        print(f'error at i = {i}')
    elif len(results[i][1][1]) != 5:
        print(f'error at i = {i}')

# %%
csv_header = ['valve_19', 'valve_22', 'valve_16', 'valve_27', 'valve_31', 'flow_35', 'flow_33', 'flow_5', 'flow_18', 'flow_9',
            'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'pressure_8']
results_array = [[None] * len(csv_header) for _ in range(len(results))]
for i in range(len(results)):
    results_array[i][0] = results[i][0][0]
    results_array[i][1] = results[i][0][1]
    results_array[i][2] = results[i][0][2]
    results_array[i][3] = results[i][0][3]
    results_array[i][4] = results[i][0][4]
    results_array[i][5] = results[i][1][0][0]
    results_array[i][6] = results[i][1][0][1]
    results_array[i][7] = results[i][1][0][2]
    results_array[i][8] = results[i][1][0][3]
    results_array[i][9] = results[i][1][0][4]
    results_array[i][10] = results[i][1][1][0]
    results_array[i][11] = results[i][1][1][1]
    results_array[i][12] = results[i][1][1][2]
    results_array[i][13] = results[i][1][1][3]
    results_array[i][14] = results[i][1][1][-1]

    output_csv_file = "simulation_results.csv"

    # Save results_array to a CSV file
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(csv_header)
        # Write the data rows
        writer.writerows(results_array)

    print(f"Results saved to {output_csv_file}")
# %%
