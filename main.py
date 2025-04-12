#%%
import wntr
import numpy as np

inp_file = 'tabletopmodel.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

#%%
diameter = 26  # mm

for pipe in wn.pipe_name_list:
    pipe = wn.get_link(pipe)
    pipe.diameter = diameter  # mm
#%%
# for valve in wn.valve_name_list:
#     valve = wn.get_link(valve)
#     # valve.diameter = diameter  # mm
#     # print(valve)    
#     wn.remove_link(valve.name)
#     wn.add_pipe(valve.name, valve.start_node_name, valve.end_node_name,
#              diameter=diameter, length=0.001, roughness=140)



#%%
def run_simulation(wn, hours=1):
    # wn.options.time.duration = hours * 3600
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    flow = results.link['flowrate'].loc[hours*3600, :]    
    return flow['1']

num_of_values = 10
valve_19_values = np.linspace(0.79, 1.54, num_of_values)
valve_22_values = np.linspace(0.46, 1.52, num_of_values)
valve_16_values = np.linspace(0.35, 1.50, num_of_values)
valve_27_values = np.linspace(0.74, 1.54, num_of_values)
valve_31_values = np.linspace(0.78, 1.56, num_of_values)

hours = 1
wn.options.time.duration = hours * 3600
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

iteration = 0
flow = run_simulation(wn, hours)
# for v19 in valve_19_values:
#     for v22 in valve_22_values:
#         for v16 in valve_16_values:
#             for v27 in valve_27_values:
#                 for v31 in valve_31_values:
#                     wn.get_link('19').initial_setting = v19
#                     wn.get_link('22').initial_setting = v22
#                     wn.get_link('16').initial_setting = v16
#                     wn.get_link('27').initial_setting = v27
#                     wn.get_link('31').initial_setting = v31

#                     flow = run_simulation(wn, hours)
#                     iteration += 1
#         print(f'Iteration {iteration}')
#%%
# pressure = results.node['pressure'].loc[hours*3600, :]
# flow = results.link['flowrate'].loc[hours*3600, :]
# wntr.graphics.plot_network(wn, node_attribute=pressure, #link_attribute=flow,
#                            node_size=150, title=f'Pressure at {hours} hours',
#                            link_labels=True, node_labels=False,
#                            node_colorbar_label='Pressure (m)', #link_colorbar_label='Flowrate (m3/s)',
#                            )
# print(pressure['2'])



# %%
