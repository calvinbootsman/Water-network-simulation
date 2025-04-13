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

# these corresponds with where the flow and pressure sensors are located
measurable_nodes = ['22', '15', '4', '3', '8']  
measureble_links = ['35', '33', '5', '18', '9']

valve_ids_to_modify = ['19', '22', '16', '27', '31', '39']
valve_setting_options = [0.01, 0.2, 1.0, 8, 50, 200, 'closed'] # Valve open: 100%, 80%, 60%, 40%, 20%, 10%, 0%

def run_simulation(wn, hours=1):
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    flows = results.link['flowrate'].loc[hours*3600, :]    
    pressures = results.node['pressure'].loc[hours*3600, :]
    flow_results = np.array([flows[x] for x in measureble_links])
    pressure_results = np.array([pressures[x] for x in measurable_nodes])
    results = np.array([flow_results, pressure_results])
    return results

# --- Worker Function ---
# This function will be executed by each process in the pool.
# It takes a single combination of settings as input.
def run_simulation_for_combination(settings_combination, hours=1):
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
        wn.reset_initial_values()
        # 2. Apply the settings for this specific combination
        for valve_id, setting in zip(valve_ids_to_modify, settings_combination):
            if setting == 'closed':
                wn.get_link(valve_id).initial_status = wntr.network.LinkStatus.Closed
            else:
                # Ensure the link is treated as open if not 'closed'
                wn.get_link(valve_id).initial_status = wntr.network.LinkStatus.Active # Or relevant Open status
                wn.get_link(valve_id).initial_setting = setting

        # 3. Run the simulation
        result = run_simulation(wn, hours)
        # 4. Return the input settings along with the result for tracking
        return (settings_combination, result)

    except Exception as e:
        # Handle potential errors during simulation gracefully
        print(f"Error processing combination {settings_combination}: {e}")
        return (settings_combination, None) # Indicate failure

# 1. Generate all combinations using itertools.product
def main(num_processes=1, hours=1):
    all_combinations = list(itertools.product(valve_setting_options, repeat=len(valve_ids_to_modify)))
    total_simulations = len(all_combinations)
    print(f"Generated {total_simulations} simulation combinations.")

    # 2. Create the pool and map the worker function to the combinations
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map applies 'run_simulation_for_combination' to each item
        # in 'all_combinations' and distributes the work across processes.
        # It blocks until all results are back.
        results = []
        for i, result in enumerate(pool.imap_unordered(run_simulation_for_combination, all_combinations, hours), start=1):
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
        else:
            failed_count += 1
            print(f"Combination {i+1}/{total_simulations}: Settings={settings} -> FAILED")

    print("\n--- Summary ---")
    print(f"Total simulations attempted: {total_simulations}")
    print(f"Successful simulations: {successful_count}")
    print(f"Failed simulations: {failed_count}")
    return results

#%%
results = []
if __name__ == "__main__":
    # Determine number of processes (leave one core free if possible)
    try:
        num_cores = os.cpu_count()
        num_processes = max(1, num_cores - 1) if num_cores else 1
    except NotImplementedError:
        num_processes = 1 # Default to 1 if cpu_count() is not implemented
    print(f"Using {num_processes} worker processes.")
    results = main(num_processes)

#%%
    # Save results to CSV
    csv_header = ['valve_19', 'valve_22', 'valve_16', 'valve_27', 'valve_31', 'valve_39','flow_35', 'flow_33', 'flow_5', 'flow_18', 'flow_9',
                'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'pressure_8']
    results_array = [[None] * len(csv_header) for _ in range(len(results))]
    for i in range(len(results)):
        results_array[i][0] = results[i][0][0]
        results_array[i][1] = results[i][0][1]
        results_array[i][2] = results[i][0][2]
        results_array[i][3] = results[i][0][3]
        results_array[i][4] = results[i][0][4]
        results_array[i][5] = results[i][0][5]
        results_array[i][6] = results[i][1][0][0] * 1000 
        results_array[i][7] = results[i][1][0][1] * 1000
        results_array[i][8] = results[i][1][0][2] * 1000
        results_array[i][9] = results[i][1][0][3] * 1000
        results_array[i][10] = results[i][1][0][4] * 1000
        results_array[i][11] = results[i][1][1][0]
        results_array[i][12] = results[i][1][1][1]
        results_array[i][13] = results[i][1][1][2]
        results_array[i][14] = results[i][1][1][3]
        results_array[i][15] = results[i][1][1][-1]

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
