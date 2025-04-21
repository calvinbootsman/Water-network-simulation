import wntr
import numpy as np
import copy
import multiprocessing
import os
import itertools
import csv
from datetime import datetime
import time
import argparse

class DatasetGenerator:
    """
    Generates a dataset of water network simulation results based on varying
    valve settings and node demands.

    Uses the WNTR (Water Network Tool for Resilience) library to run hydraulic
    simulations for numerous parameter combinations, potentially in parallel.
    The results are saved to a CSV file.

    Attributes:
        wn (wntr.network.WaterNetworkModel): A deep copy of the initial water network model.
        output_dir (str): Directory where the dataset CSV and combinations file will be saved.
        output_file_path (str): Full path to the output dataset CSV file.
        combinations_file_path (str): Full path to the file storing parameter combinations.
        measurable_nodes (list): List of node IDs whose pressure will be recorded.
        measureble_links (list): List of link IDs whose flowrate will be recorded.
        valve_ids_to_modify (list): List of valve IDs whose settings will be varied.
        valve_setting_options (list): Discrete settings to apply to valves, including 'closed'.
        node_ids_to_modify (list): List of node IDs whose base demand will be varied.
        base_demand_values (list): Discrete base demand values to apply to nodes.
        ids_to_modify (list): Combined list of valve and node IDs prefixed with 'v' or 'n'.
        csv_header (list): Header row for the output CSV file.
    """
    def __init__(self, wn, output_dir, combinations_filename="all_combinations.csv"):
        """
        Initializes the DatasetGenerator.

        Args:
            wn (wntr.network.WaterNetworkModel): The initial water network model.
            output_dir (str): The directory to save output files.
            combinations_filename (str): The name of the file to save/load combinations.
        """
        self.output_dir = output_dir
        self.wn = copy.deepcopy(wn) # Create a deep copy to avoid modifying the original model

        # Define parameters and measurable outputs
        # Valves to be adjusted
        self.valve_ids_to_modify = ['19', '22', '16', '27', '31', '39']
        # Possible settings for each valve (0.01 to 50 are settings, 'closed' means status=closed)
        self.valve_setting_options = [0.01, 0.2, 1.0, 8, 50, 'closed']

        # Nodes whose demand will be adjusted
        self.node_ids_to_modify = ['27']
        # Possible base demand values for each node
        self.base_demand_values = [0.0, 0.3, 0.6, 0.9, 1.2]

        # Combine valve and node IDs with prefixes for easier identification
        self.ids_to_modify = [f'v{x}' for x in self.valve_ids_to_modify] + \
                             [f'n{x}' for x in self.node_ids_to_modify]

        # Nodes and links where measurements (results) will be taken
        self.measurable_nodes = ['22', '15', '4', '3', '8']
        self.measureble_links = ['35', '33', '5', '18', '9']

        # Define the header for the output CSV file
        # Input parameters first, then output measurements
        self.csv_header = [f'valve_{v}' for v in self.valve_ids_to_modify] + \
                          [f'node_{n}' for n in self.node_ids_to_modify] + \
                          [f'flow_{l}' for l in self.measureble_links] + \
                          [f'pressure_{n}' for n in self.measurable_nodes]

        # --- File Handling ---
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        # Define the full path for the combinations file
        self.combinations_file_path = os.path.join(self.output_dir, combinations_filename)

        # Define the full path for the main dataset output file
        # Include a timestamp to avoid overwriting previous runs
        dataset_filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.output_file_path = os.path.join(self.output_dir, dataset_filename)
        print(f"Output dataset will be saved to: {self.output_file_path}")
        print(f"Combinations will be saved/loaded from: {self.combinations_file_path}")


    def _run_simulation(self, wn_modified=None):
        """
        Runs a single WNTR simulation for the specified duration.

        Args:
            wn_modified (wntr.network.WaterNetworkModel, optional): The water network model
                to simulate. If None, uses the instance's base model. Defaults to None.

        Returns:
            numpy.ndarray: A 2D numpy array containing flow results (row 0, in m^3/s) and
                           pressure results (row 1, in m). Returns None if simulation fails.
            bool: True if negative pressures were detected, False otherwise.
        """
        if wn_modified is None:
            wn_modified = self.wn # Use the base model if no modified one is provided

        try:
            # Configure and run the WNTR simulator
            sim = wntr.sim.WNTRSimulator(wn_modified)
            # Increased MAXITER for potentially complex scenarios
            results = sim.run_sim(solver_options={'MAXITER': 300_000})

            # Extract flowrates and pressures at the specified time point (in seconds)
            # WNTR results are indexed by time in seconds
            time_in_seconds = 3600
            flows = results.link['flowrate'].loc[time_in_seconds, :]
            pressures = results.node['pressure'].loc[time_in_seconds, :]

            # Get results for the specific links and nodes we care about
            flow_results = np.array([flows[link_id] for link_id in self.measureble_links])
            pressure_results = np.array([pressures[node_id] for node_id in self.measurable_nodes])

            # Combine flow and pressure results into a single array
            # Note: Flow rates are in m^3/s, Pressures are in m
            flattened_results = np.array([flow_results, pressure_results])
            return flattened_results

        except Exception as e:
            # Catch potential errors during simulation (e.g., convergence issues)
            print(f"Simulation failed: {e}")
            # Optional: Log the specific error and parameters if needed
            return None # Indicate failure

    def _run_simulation_for_combination(self, settings_combination):
        """
        Creates a copy of the network, applies a specific combination of settings,
        runs the simulation, and returns the settings along with the results.

        Designed to be used with multiprocessing.Pool.imap_unordered.

        Args:
            settings_combination (tuple): A tuple of settings corresponding to the
                                          elements in `self.ids_to_modify`.

        Returns:
            tuple: A tuple containing (settings_combination, result_array).
                   result_array is a numpy array from _run_simulation, or None if
                   the simulation failed or had negative pressure.
        """
        # Create a fresh copy of the base network for this specific simulation
        # This is crucial for parallel execution to avoid race conditions
        wn_copy = copy.deepcopy(self.wn)

        try:
            # Apply the settings from the combination to the network copy
            for i, setting in enumerate(settings_combination):
                element_id_with_prefix = self.ids_to_modify[i]
                element_type = element_id_with_prefix[0] # 'v' for valve, 'n' for node
                element_id = element_id_with_prefix[1:]  # The actual ID number as a string

                if element_type == 'v': # Apply valve setting
                    valve = wn_copy.get_link(element_id)
                    if setting == 'closed':
                        # Set the valve status to Closed
                        valve.initial_status = wntr.network.LinkStatus.Closed
                        # Setting value might not matter when closed, but set to 0 for clarity
                        valve.initial_setting = 0
                    else:
                        # Ensure the valve is treated as Active (Open)
                        valve.initial_status = wntr.network.LinkStatus.Active
                        # Apply the numerical setting (e.g., pressure drop coefficient)
                        valve.initial_setting = float(setting)
                elif element_type == 'n': # Apply node demand setting
                    node = wn_copy.get_node(element_id)
                    # Assume demand is applied via the first timeseries pattern
                    # Modify the base value of that pattern
                    if node.demand_timeseries_list:
                         node.demand_timeseries_list[0].base_value = float(setting)
                    else:
                        # Handle cases where the node might not have a demand pattern initially
                        # This might involve adding a pattern, depending on requirements.
                        print(f"Warning: Node {element_id} has no demand timeseries list. Cannot set demand.")


            # Run the simulation with the modified network copy
            result = self._run_simulation(wn_modified=wn_copy)

            # Return the input settings along with the simulation results (or None)
            return (settings_combination, result)

        except Exception as e:
            # Catch errors during network modification or simulation setup
            print(f"Error processing combination {settings_combination}: {e}")
            # Return the combination and None for the result to indicate failure
            return (settings_combination, None, False)

    def get_combinations(self, num_random_combinations=1_000_000, load_if_exists=True, save_combinations=True):
        """
        Generates or loads a list of parameter combinations to simulate.

        Includes combinations from the Cartesian product of discrete options,
        plus a specified number of randomly generated combinations within ranges.

        Args:
            num_random_combinations (int): Number of random combinations to generate and add.
            load_if_exists (bool): If True and the combinations file exists, load from it.
            save_combinations (bool): If True, save the generated combinations to the file.

        Returns:
            list: A list of tuples, where each tuple represents a combination of settings
                  corresponding to `self.ids_to_modify`.
        """
        # Check if a pre-existing combinations file should be loaded
        if load_if_exists and os.path.exists(self.combinations_file_path):
            print(f"Loading combinations from {self.combinations_file_path}...")
            all_combinations = []
            try:
                with open(self.combinations_file_path, mode='r', newline='') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        # Convert numerical strings back to floats, keep 'closed' as string
                        processed_row = []
                        for item in row:
                            try:
                                # Attempt to convert to float
                                processed_row.append(float(item))
                            except ValueError:
                                # Keep as string if conversion fails (e.g., 'closed')
                                processed_row.append(item)
                        all_combinations.append(tuple(processed_row))
                print(f"Loaded {len(all_combinations)} combinations.")
                return all_combinations
            except Exception as e:
                print(f"Error loading combinations file: {e}. Will generate new combinations.")

        print("Generating new combinations...")
        # --- Generate Combinations based on Discrete Options ---
        # 1. Cartesian product of discrete valve settings
        all_valve_combinations = list(itertools.product(self.valve_setting_options,
                                                        repeat=len(self.valve_ids_to_modify)))
        # 2. Cartesian product of discrete node demand settings
        all_node_combinations = list(itertools.product(self.base_demand_values,
                                                       repeat=len(self.node_ids_to_modify)))
        # 3. Combine valve and node discrete combinations
        #    This creates combinations where all valves have a discrete setting AND
        #    all nodes have a discrete setting.
        discrete_combinations = []
        if all_valve_combinations and all_node_combinations:
             # Use itertools.product to combine the two sets of combinations
            for valve_comb, node_comb in itertools.product(all_valve_combinations, all_node_combinations):
                 discrete_combinations.append(valve_comb + node_comb)
        elif all_valve_combinations: # Only valves defined
            discrete_combinations.extend(all_valve_combinations)
        elif all_node_combinations: # Only nodes defined
             discrete_combinations.extend(all_node_combinations)

        print(f"Generated {len(discrete_combinations)} discrete combinations.")

        # --- Generate Random Combinations ---
        randomly_generated_combinations = []
        print(f"Generating {num_random_combinations} random combinations...")
        num_valves = len(self.valve_ids_to_modify)
        num_nodes = len(self.node_ids_to_modify)
        valve_options_numeric = [opt for opt in self.valve_setting_options if opt != 'closed'] # Exclude 'closed' for range calculation

        for _ in range(num_random_combinations):
            new_settings = []
            # Random settings for valves
            for _ in range(num_valves):
                # Small chance to explicitly set valve to 'closed'
                if np.random.rand() < 0.05: # 5% chance of being closed
                    new_settings.append('closed')
                else:
                    # Choose a random interval between discrete numeric options
                    # This creates more varied settings than just the discrete ones
                    idx = np.random.randint(0, len(valve_options_numeric) -1) # Index for start value
                    start_val = valve_options_numeric[idx]
                    end_val = valve_options_numeric[idx + 1]
                    # Handle edge case for the last interval if needed (optional extension)
                    # if idx == len(valve_options_numeric) - 2:
                    #     end_val = max(valve_options_numeric) * 1.5 # Example: extend range slightly

                    # Generate a random float within the chosen interval
                    new_settings.append(np.random.uniform(start_val, end_val))

            # Random settings for node demands
            for _ in range(num_nodes):
                 # Generate random demand between min and max specified base demands
                 min_demand = min(self.base_demand_values)
                 max_demand = max(self.base_demand_values)
                 new_settings.append(np.random.uniform(min_demand, max_demand))

            randomly_generated_combinations.append(tuple(new_settings))

        # Combine discrete and random combinations
        all_combinations = discrete_combinations + randomly_generated_combinations
        total_simulations = len(all_combinations)
        print(f"Total generated combinations: {total_simulations}")

        # Save the generated combinations if requested
        if save_combinations:
            try:
                print(f"Saving {total_simulations} combinations to {self.combinations_file_path}...")
                with open(self.combinations_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(all_combinations)
                print("Combinations saved successfully.")
            except Exception as e:
                print(f"Error saving combinations file: {e}")

        return all_combinations

    def generate(self, num_random_combinations=1_000_000, num_processes=None, checkpoint_interval=0, load_combinations=True, save_combinations=True):
        """
        Generates the dataset by running simulations for multiple combinations in parallel.

        Args:
            num_random_combinations (int, optional): Number of random combinations to add.
                                                      Defaults to 1,000,000.
            num_processes (int, optional): Number of parallel processes to use.
                                           Defaults to os.cpu_count() - 1.
            checkpoint_interval (int, optional): Save results every N simulations.
                                                 If 0, saves only at the end. Defaults to 0.
            load_combinations (bool, optional): Load combinations from file if it exists.
                                                Defaults to True.
            save_combinations (bool, optional): Save generated combinations to file.
                                                Defaults to True.
        """
        # Get the list of all simulation parameter combinations
        all_combinations = self.get_combinations(
            num_random_combinations=num_random_combinations,
            load_if_exists=load_combinations,
            save_combinations=save_combinations
        )
        total_simulations = len(all_combinations)
        if total_simulations == 0:
            print("No combinations to simulate. Exiting.")
            return

        # Determine the number of processes for parallel execution
        if num_processes is None:
            try:
                num_cores = os.cpu_count()
                # Default to N-1 cores, or 1 if only 1 core is available
                num_processes = max(1, num_cores - 1) if num_cores else 1
            except NotImplementedError:
                num_processes = 1 # Fallback if cpu_count() is not available
        print(f"Running simulations using {num_processes} processes.")

        results_buffer = [] # Temporary storage for results before saving
        processed_count = 0
        start_time = time.time()
        last_print_time = start_time

        # Use multiprocessing Pool for parallel execution
        # imap_unordered processes tasks as they complete, good for progress tracking
        # We pass _run_simulation_for_combination (instance method) and the list of combinations
        with multiprocessing.Pool(processes=num_processes) as pool:
            # The second argument to `itertools.repeat` passes the 'hours' argument
            # to each call of `_run_simulation_for_combination`
            simulation_tasks = pool.imap_unordered(
                self._run_simulation_for_combination,
                all_combinations,
                chunksize=max(1, total_simulations // (num_processes * 4000)) # Adjust chunksize for efficiency
            )

            # Process results as they become available
            for i, result_tuple in enumerate(simulation_tasks, start=1):
                processed_count += 1
                settings, sim_result = result_tuple

                if sim_result is not None: # Only add successful simulations to buffer
                     results_buffer.append((settings, sim_result))


                # Print progress periodically
                current_time = time.time()
                if current_time - last_print_time >= 30.0 or processed_count == total_simulations: # Print every 60 seconds or at the end
                     elapsed_total = current_time - start_time
                     avg_time_per_sim = elapsed_total / processed_count if processed_count else 0
                     print(f"Processed: {processed_count}/{total_simulations} ({processed_count/total_simulations*100:.1f}%) | "
                           f"Avg time/sim: {avg_time_per_sim:.3f}s | Total time: {elapsed_total:.1f}s")
                     last_print_time = current_time


                # Checkpoint saving
                # Save if checkpoint interval is reached AND results_buffer is not empty
                if checkpoint_interval > 0 and processed_count % checkpoint_interval == 0 and results_buffer:
                    self._save_results(results_buffer)
                    results_buffer = [] # Clear the buffer after saving

        # Final save for any remaining results in the buffer
        if results_buffer:
            print(f"\nSaving remaining {len(results_buffer)} results...")
            self._save_results(results_buffer)
            print("Final save complete.")
        else:
             print("\nNo results remaining in buffer to save.")


        total_time = time.time() - start_time
        print(f"\n--- Simulation Complete ---")
        print(f"Total simulations processed: {processed_count}")
        print(f"Total successful results saved: Check CSV file row count (excluding header).") # More reliable than summing buffer lengths
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Dataset saved to: {self.output_file_path}")
        # Update the instance counter (note: this won't be accurate with multiprocessing unless using shared memory/manager)


    def _save_results(self, results_list):
        """
        Saves a list of simulation results to the dataset CSV file. Appends if the file exists.

        Args:
            results_list (list): A list of tuples, where each tuple is
                                 (settings_combination, simulation_result_array).
                                 simulation_result_array is expected to be the 2D numpy
                                 array from _run_simulation ([flows], [pressures]).
        """
        if not results_list:
             print("No results to save.")
             return

        # Check if the file already exists to determine if the header should be written
        file_exists = os.path.exists(self.output_file_path)

        try:
            with open(self.output_file_path, mode='a', newline='') as file: # Open in append mode ('a')
                writer = csv.writer(file)

                # Write the header row ONLY if the file is new
                if not file_exists:
                    writer.writerow(self.csv_header)

                # Process and write each result row
                rows_to_write = []
                for settings, result_array in results_list:
                     # Ensure result_array is valid (not None) before processing
                     if result_array is None or result_array.shape != (2, len(self.measureble_links)) or result_array.shape[1] != len(self.measurable_nodes):
                         # Handle cases where simulation might have failed silently or returned unexpected shape
                         # print(f"Skipping invalid result data for settings: {settings}") # Optional: Log skipped data
                         continue # Skip this row

                     try:
                         # Extract flows (m^3/s -> L/s by multiplying by 1000) and pressures (m)
                         # Ensure we access the correct indices based on _run_simulation output structure
                         flows_L_per_s = result_array[0] * 1000
                         pressures_m = result_array[1]

                         # Combine settings and results into a single list for the CSV row
                         # Order must match self.csv_header
                         row_data = list(settings) + list(flows_L_per_s) + list(pressures_m)
                         rows_to_write.append(row_data)

                     except IndexError as ie:
                          print(f"Error processing result data shape for settings {settings}: {ie}. Skipping row.")
                     except Exception as e:
                          print(f"Unexpected error processing result for settings {settings}: {e}. Skipping row.")


                # Write all processed rows to the CSV file
                if rows_to_write:
                     writer.writerows(rows_to_write)


        except IOError as e:
            print(f"Error writing to CSV file {self.output_file_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during saving: {e}")


    def test_single_simulation(self, settings_combination_str):
        """
        Runs and prints the result for a single, specific combination provided as a string.

        Useful for debugging specific scenarios.

        Args:
            settings_combination_str (str): A comma-separated string of settings
                                           (e.g., "1.0,closed,0.5,8,...").

        Returns:
             tuple: Result from _run_simulation_for_combination (settings, result_array, had_negative)
                    or None if input parsing fails.
        """
        try:
            # Convert the comma-separated string into a tuple of floats/strings
            settings_list = settings_combination_str.split(',')
            processed_settings = []
            for item in settings_list:
                item = item.strip() # Remove leading/trailing whitespace
                try:
                    processed_settings.append(float(item))
                except ValueError:
                    if item.lower() == 'closed':
                         processed_settings.append('closed')
                    else:
                         # Handle unexpected string values if necessary
                         print(f"Warning: Unknown setting value '{item}' in test string.")
                         processed_settings.append(item) # Keep original string if not float or 'closed'

            settings_tuple = tuple(processed_settings)

            # Verify the number of settings matches expected
            if len(settings_tuple) != len(self.ids_to_modify):
                 print(f"Error: Provided settings count ({len(settings_tuple)}) does not match "
                       f"expected count ({len(self.ids_to_modify)}).")
                 print(f"Expected order: {self.ids_to_modify}")
                 return None


            print(f"Testing simulation with settings: {settings_tuple}")
            result = self._run_simulation_for_combination(settings_tuple)
            print("-" * 20)
            print(f"Input Settings: {result[0]}")
            if result[1] is not None:
                print(f"Flows (L/s): {result[1][0] * 1000}")
                print(f"Pressures (m): {result[1][1]}")
                print(f"Negative Pressure Detected: {result[2]}")
            else:
                print("Simulation Failed or Resulted in Negative Pressure (result is None).")
                print(f"Negative Pressure Detected Flag: {result[2]}")
            print("-" * 20)
            return result

        except Exception as e:
            print(f"Error during single test simulation: {e}")
            return None


# Main execution block when the script is run directly
if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Water Network Simulation Dataset.")

    # Required arguments
    parser.add_argument("inp_file", help="Path to the WNTR input file (.inp).")

    # Optional arguments
    parser.add_argument("-o", "--output_dir", default="./dataset_output",
                        help="Directory to save the dataset and combinations file (default: ./dataset_output).")
    parser.add_argument("-c", "--combinations_file", default="simulation_combinations.csv",
                        help="Filename for saving/loading parameter combinations (default: simulation_combinations.csv).")
    parser.add_argument("-r", "--random_combinations", type=int, default=10000,
                        help="Number of random combinations to generate (default: 10000).")
    parser.add_argument("-p", "--num_processes", type=int, default=None,
                        help="Number of parallel processes (default: auto-detect cores - 1).")
    parser.add_argument("-k", "--checkpoint", type=int, default=1000,
                        help="Checkpoint interval: save results every N simulations (0 to disable, default: 1000).")
    parser.add_argument("--no_load", action="store_true",
                        help="Force regeneration of combinations, do not load existing file.")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save the generated combinations file.")
    parser.add_argument("--leak_node", type=str, default=None,
                        help="Node ID to add a leak to (e.g., '2').")
    parser.add_argument("--leak_area", type=float, default=0.05,
                        help="Area for the leak if --leak_node is specified (default: 0.05).")
    parser.add_argument("--test_single", type=str, default=None,
                        help="Run only a single test simulation with the provided comma-separated settings string.")


    args = parser.parse_args()

    # --- Initial Water Network Setup ---
    print(f"Loading water network model from: {args.inp_file}")
    try:
        wn_initial = wntr.network.WaterNetworkModel(args.inp_file)
        # Set demand model to Pressure Dependent Demand (PDD)
        wn_initial.options.hydraulic.demand_model = 'PDD'
        print("Water network model loaded successfully.")

        # Optional: Add a leak based on command-line arguments
        node = wn_initial.get_node('2')
        #  surface area of the pipe  = (0.26/2)^2 * pi 
        node.add_leak(wn_initial, area = 0.053)
        active_control_action = wntr.network.ControlAction(node, "leak_status", True)
        control = wntr.network.controls.Control._time_control(
                    wn_initial, 0, "SIM_TIME", False, active_control_action
                )
        wn_initial.add_control("control1", control)


    except FileNotFoundError:
        print(f"Error: Input file not found at {args.inp_file}")
        exit(1) # Exit the script if the input file is not found
    except Exception as e:
        print(f"Error loading water network model: {e}")
        exit(1) # Exit on other loading errors

    # --- Initialize and Run Dataset Generator ---
    print("Initializing Dataset Generator...")
    dg = DatasetGenerator(wn_initial,
                          output_dir=args.output_dir,
                          combinations_filename=args.combinations_file)

    if args.test_single:
        # Run only a single test case if specified
        dg.test_single_simulation(args.test_single)
    else:
        # Run the full dataset generation process
        print("Starting dataset generation...")
        dg.generate(
            num_random_combinations=args.random_combinations,
            num_processes=args.num_processes,
            checkpoint_interval=args.checkpoint,
            load_combinations=(not args.no_load), # Invert the flag logic
            save_combinations=(not args.no_save)  # Invert the flag logic
        )

    print("Script finished.")