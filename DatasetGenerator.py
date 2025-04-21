import wntr
import numpy as np
import threading
import copy
import multiprocessing
import os
import itertools
import csv
from datetime import datetime
import time
class DatasetGenerator:
    def __init__(self, wn, output_dir):
        self.output_dir = output_dir
        self.wn = copy.deepcopy(wn)
        csv_header = ['valve_19', 'valve_22', 'valve_16', 'valve_27', 'valve_31', 'valve_39','node_27',
                    'flow_35', 'flow_33', 'flow_5', 'flow_18', 'flow_9',
                    'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'pressure_8']
        
        self.measurable_nodes = ['22', '15', '4', '3', '8']  
        self.measureble_links = ['35', '33', '5', '18', '9']

        self.valve_ids_to_modify = ['19', '22', '16', '27', '31', '39']
        self.valve_setting_options = [0.01, 0.2, 1.0, 8, 50, 'closed'] # Valve open: 100%, 80%, 60%, 40%, 20%, 0%

        self.node_ids_to_modify = ['27']
        self.base_demand_values = [0.0, 0.3, 0.6, 0.9, 1.2]
        self.ids_to_modify = self.valve_ids_to_modify + self.node_ids_to_modify

        self.ids_to_modify = ['v'+x for x in self.valve_ids_to_modify] + ['n'+x for x in self.node_ids_to_modify]
        self.negative_values_counter = 0
        
        self.file_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.output_dir = './dataset'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file_path = os.path.join(self.output_dir, self.file_name)

        
    def _run_simulation(self, hours=2, wn=None):
        if wn is None:
            wn = self.wn
        sim = wntr.sim.WNTRSimulator(wn)
        
        results = sim.run_sim(solver_options={'MAXITER': 300_000})
        flows = results.link['flowrate'].loc[hours*3600, :]    
        pressures = results.node['pressure'].loc[hours*3600, :]
        flow_results = np.array([flows[x] for x in self.measureble_links])
        pressure_results = np.array([pressures[x] for x in self.measurable_nodes])

        # if np.any(pressure_results < 0):
        #     self.negative_values
        #     negative_values += 1
        flatten_results = np.array([flow_results, pressure_results])

        return flatten_results
    
    def _run_simulation_for_combination(self, settings_combination, hours=1):
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
            wn = copy.deepcopy(self.wn)
            for id, setting in zip(self.ids_to_modify, settings_combination):
                id_number = id[1:]
                if id[0] == 'v':                
                    if setting == 'closed':
                        wn.get_link(id_number).initial_status = wntr.network.LinkStatus.Closed
                    else:
                        # Ensure the link is treated as open if not 'closed'
                        wn.get_link(id_number).initial_status = wntr.network.LinkStatus.Active
                        wn.get_link(id_number).initial_setting = setting
                elif id[0] == 'n':
                    wn.get_node(id_number).demand_timeseries_list[0].base_value = setting

            result = self._run_simulation(hours, copy.deepcopy(wn))

            return (settings_combination, result)

        except Exception as e:
            print(f"Error processing combination {settings_combination}: {e}")
            return (settings_combination, None)
        
    def get_combinations(self, random_combinations=1_000_000, load_from_file=True):
        if load_from_file:
            if os.path.exists("all_combinations.csv"):
                print("all_combinations.csv already exists. Loading combinations from file.")
                with open("all_combinations.csv", mode='r') as file:
                    reader = csv.reader(file)
                    all_combinations = []
                    for row in reader:
                        row = [float(x) if x.replace('.', '', 1).isdigit() else x for x in row]
                        all_combinations.append(tuple(row))
                return all_combinations
        
        all_valve_combinations = list(itertools.product(self.valve_setting_options, repeat=len(self.valve_ids_to_modify)))
        all_node_combinations = list(itertools.product(self.base_demand_values, repeat=len(self.node_ids_to_modify)))
        all_combinations = list(itertools.product(all_valve_combinations, all_node_combinations))
        all_combinations = [valve_comb + node_comb for valve_comb, node_comb in all_combinations]

        additional_combinations = []
        for i in range(random_combinations):
            new_settings = []
            for j in range(len(self.ids_to_modify)):    
                if j < len(self.valve_ids_to_modify):
                    if np.random.rand() < 0.1:
                        new_settings.append('closed')
                    else:
                        index = np.random.randint(0, len(self.valve_setting_options)-2)
                        start_value = self.valve_setting_options[index]
                        end_value = self.valve_setting_options[index + 1]
                        if index == len(self.valve_setting_options)-2:
                            end_value = 250
                        new_settings.append(np.random.uniform(start_value, end_value))    
                else:
                    start_value = self.base_demand_values[0]
                    end_value = self.base_demand_values[-1]   
                    
                    new_settings.append(np.random.uniform(start_value, end_value))
            additional_combinations.append(tuple(new_settings))
        all_combinations += additional_combinations

        total_simulations = len(all_combinations)

        print(f"Generated {total_simulations} simulation combinations.")
        # Save all_combinations to a CSV file
        output_combinations_file = "all_combinations.csv"
        with open(output_combinations_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write each combination as a row
            writer.writerows(all_combinations)

        print(f"All combinations saved to {output_combinations_file}")
        return all_combinations
    
    def generate(self, random_combinations=1_000_000, hours=1, num_processes=None, checkpoint=0, load_from_file=True):
        """
        Generates a dataset of simulation results based on valve settings and node demands.

        Args:
            random_combinations (int): Number of random combinations to generate.
            hours (int): Duration of the simulation in hours.

        Returns:
            list: A list of tuples containing (settings_combination, result).
        """
        all_combinations = self.get_combinations(random_combinations=random_combinations, load_from_file=load_from_file)
        total_simulations = len(all_combinations)

        if num_processes is None:
            try:
                num_cores = os.cpu_count()
                num_processes = max(1, num_cores - 1) if num_cores else 1
            except NotImplementedError:
                num_processes = 1 # Default to 1 if cpu_count() is not implemented

        results = []
        last_print_time = time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i, result in enumerate(pool.imap_unordered(self._run_simulation_for_combination, all_combinations, hours), start=1):
                if i%1000 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - last_print_time
                    last_print_time = current_time
                    print(f"Processed combination {i}/{total_simulations}, Elapsed time: {elapsed_time:.2f} seconds")
                results.append(result)
                
                if checkpoint > 0:
                    if i % checkpoint == 0:
                        self._save_results(copy.deepcopy(results))
                        results = []
        self._save_results(results)
        print(f"Total negative values encountered: {self.negative_values_counter}")

    def test_single_simulation(self, settings_combination, hours=1):
        """
        Test a single simulation with the given settings combination.

        Args:
            settings_combination (tuple): A tuple of settings corresponding
                                        to valve_ids_to_modify.

        Returns:
            tuple: A tuple containing (settings_combination, result) or
                (settings_combination, None) if simulation failed.
        """

        settings_combination = tuple(float(x) if x.replace('.', '', 1).isdigit() else x for x in settings_combination.split(','))
        return self._run_simulation_for_combination(settings_combination, hours=hours)

    def _save_results(self, results):
        """
        Save the results to a CSV file.

        Args:
            results (list): List of results to save.
            i (int): Current iteration number.
            checkpoint (int): Checkpoint interval.
        """
        csv_header = ['valve_19', 'valve_22', 'valve_16', 'valve_27', 'valve_31', 'valve_39','node_27',
                    'flow_35', 'flow_33', 'flow_5', 'flow_18', 'flow_9',
                    'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'pressure_8']
        results_array = [[None] * len(csv_header) for _ in range(len(results))]

        for i in range(len(results)):
                try:
                    new_results = [[None] for _ in range(len(csv_header))]
                    new_results[0] = results[i][0][0]
                    new_results[1] = results[i][0][1]
                    new_results[2] = results[i][0][2]
                    new_results[3] = results[i][0][3]
                    new_results[4] = results[i][0][4]
                    new_results[5] = results[i][0][5]
                    new_results[6] = results[i][0][6]
                    new_results[7] = results[i][1][0][0] * 1000 
                    new_results[8] = results[i][1][0][1] * 1000
                    new_results[9] = results[i][1][0][2] * 1000
                    new_results[10] = results[i][1][0][3] * 1000
                    new_results[11] = results[i][1][0][4] * 1000
                    new_results[12] = results[i][1][1][0]
                    new_results[13] = results[i][1][1][1]
                    new_results[14] = results[i][1][1][2]
                    new_results[15] = results[i][1][1][3]
                    new_results[16] = results[i][1][1][4]
                    results_array[i] = new_results
                except Exception as e:
                    print(f"Error processing result {i}: {e}")
                    results_array[i] = [None] * len(csv_header)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            # Save results_array to a CSV file
        file_exists = os.path.exists(self.output_file_path)
        with open(self.output_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            if not file_exists:
                writer.writerow(csv_header)
            # Write the data rows
            writer.writerows(results_array)

if __name__ == "__main__":   
    inp_file = 'tabletopmodel.inp'
    wn_inital = wntr.network.WaterNetworkModel(inp_file)
    wn_inital.options.hydraulic.demand_model = 'PDD'
    node = wn_inital.get_node('2')
    #  surface area of the pipe  = (0.26/2)^2 * pi 
    node.add_leak(wn_inital, area = 0.053)
    active_control_action = wntr.network.ControlAction(node, "leak_status", True)
    control = wntr.network.controls.Control._time_control(
                wn_inital, 0, "SIM_TIME", False, active_control_action
            )
    wn_inital.add_control("control1", control) 
    pipe = wn_inital.get_link('35')
    

    dg = DatasetGenerator(wn_inital, output_dir='./dataset')
    dg.generate(random_combinations=1_000_000, hours=1, checkpoint=10_000, load_from_file=True)