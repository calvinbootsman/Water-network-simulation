# WDN neural network
![Tabletop Model](tabletopmodel.jpg)

The epanet model currently contains:
- 20 pipes
- 21 junctions
- 2 consumers (modeled as tanks)
- 7 valves

# Neural network
- convolutional neural network
- the  4 flow and 4 pressure sensors as inputs
- tanh as activation function -> unsure about negative flows and pressure
- output at first consumer 1 (2 nodes: 1 for pressure and 1 for flow)

