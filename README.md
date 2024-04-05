# pyGRAPE

`pyGRAPE.py` is a Python code equivalent to [GRAPE](https://github.com/inmo-jang/GRAPE), which is for the paper "Anonymous Hedonic Game for Task Allocation in a Large-Scale Multiple Agent System" in IEEE T-RO ([DOI: 10.1109/TRO.2018.2858292](https://ieeexplore.ieee.org/document/8439076))



## Installation

To use `pyGRAPE.py`, ensure you have Python installed on your system. This script requires the following Python packages:

- NumPy
- Matplotlib
- ImageIO

You can install these packages using pip:

```bash
pip install numpy matplotlib imageio
```

## Usage

### Simple Testing

When you execute `pyGRAPE.py` with its main function as follows, then you will get some visualisation results. 

```bash
python pyGRAPE.py
```

The results will look like this:
![alt text](https://github.com/inmo-jang/pyGRAPE/blob/main/result_animation.gif)


### Usage Example
This section provides a more detailed example of using `pyGRAPE.py` to generate a scenario, verify connectivity, visualise the scenario, check utility, perform task allocation, and generate a process animation as a GIF.

```python
# Scenario
num_tasks = 3
num_agents = 2**5 * num_tasks
deployment_type = 'circle'
comm_distance = 130

scenario = generate_scenario(num_tasks, num_agents, comm_distance=comm_distance, deployment_type=deployment_type)
verify_scenario_connectivity(scenario['agent_comm_matrix'])
visualise_scenario(scenario, filename='fig_initial_allocation_visualization.png')

# Utility Check
agent_id = 0
task_id = 0
environment = scenario['environment']
visualise_utility(agent_id, task_id, environment, max_participants=num_agents, filename='fig_utility')

# Task Allocation
allocation_result = grape_allocation(scenario)
visualise_scenario(scenario, allocation_result['final_allocation'], filename="fig_final_allocation_visualization.png")

# Generate the process as GIF
generate_gif_from_history(scenario, allocation_result, filename='result_animation.gif')
```

This comprehensive example demonstrates the workflow from scenario generation to task allocation and visualisation of results, including generating an animated GIF of the process.



## Function Descriptions

`pyGRAPE.py` includes several functions designed to facilitate the generation, analysis, and visualisation of scenarios within multi-agent systems. Below are brief descriptions of each function:

### Scenario Generation
#### `generate_scenario(num_tasks, num_agents, comm_distance, gap_agent, gap_task, task_location_range, agent_location_range, deployment_type)`
Generates a scenario with specified parameters, creating task and agent locations, and returns a dictionary containing the generated scenario data.

#### `generate_task_or_agent_location(num_entities, limit, existing_locations, gap, is_task, deployment_type)`
Generates locations for tasks or agents ensuring they respect specified constraints like minimum distance between them, and, for agents, their deployment shape.


#### `verify_scenario_connectivity(agent_comm_matrix, flag_display=True)`
Verifies if all agents in the scenario are connected directly or indirectly using the agent communication matrix. It returns `True` if the scenario is connected, otherwise `False`.

### Task Allocation
#### `grape_allocation(scenario, display_progress)`
Executes the GRAPE task allocation algorithm to assign agents to tasks based on their utilities and returns a dictionary with the final allocations and utility information.

#### `calculate_utility(agent_id, task_id, utility_type, ignore_cost, agent_location, task_location, task_demand)`
Calculates the utility of an agent for a task, considering various factors such as distance, demand, and utility type.

#### `distributed_mutex(agents, agent_comm_matrix)`
Performs a distributed consensus (mutex) operation among agents based on their communication network, deciding based on the highest iteration number among agents or the latest timestamp in case of a tie.


### Visualisation
#### `visualise_utility(agent_id, task_id, environment, max_participants, util_types, filename)`
Visualises the change in utility value with the number of participants for various utility types, saving the visualisation to a specified file.

#### `visualise_scenario(scenario, final_allocation, filename)`
Visualises the final allocation of agents to tasks, including the communication network between agents, and saves the visualisation to a specified file.

#### `generate_gif_from_history()`
This function's specific details were not provided in the extracted information, suggesting it may generate a GIF from the allocation history or simulation steps.

## Contributing

Contributions to `pyGRAPE.py` are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License
[GNU GPLv3](https://github.com/inmo-jang/pyGRAPE/blob/main/LICENSE)
