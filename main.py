#%%
import DatasetGenerator
import wntr
import copy
import time

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
    

    dg = DatasetGenerator.DatasetGenerator(wn_inital, output_dir='./dataset')
    dg.generate(random_combinations=1_000, hours=1, checkpoint=10_000, load_from_file=True)
    # start_time = time.time()

    # # Run 1000 simulations
    # for i in range(1000):
    #     dg.test_single_simulation('closed,closed,closed,0.01,0.01,0.01,0.0001')

    # end_time = time.time()
    # print(f"Stress test completed: 1000 simulations in {end_time - start_time:.2f} seconds")

    # print(dg.test_single_simulation('0.01,0.01,0.01,0.01,0.01,50,0.8'))
    # print(dg.test_single_simulation('closed,closed,closed,0.01,0.01,0.01,0.0002'))
    # print(dg.test_single_simulation('closed,closed,closed,0.01,0.01,0.01,0.0005'))
    # print(dg.test_single_simulation('closed,closed,closed,0.01,0.01,0.01,0.0012'))

    # dg.generate(random_combinations=1_000, hours=1, num_processes=1, checkpoint=1000, load_from_file=False)

# %%
