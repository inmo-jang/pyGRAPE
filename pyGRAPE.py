import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

## Functions for Scenario Generation
def verify_scenario_connectivity(agent_comm_matrix, flag_display=True):
    """
    Verifies if all agents in the scenario are connected directly or indirectly.
    
    Parameters:
    - agent_comm_matrix (np.ndarray): The adjacency matrix representing agent communications.
    - flag_display (bool): Whether to print the connectivity status to the console.
    
    Returns:
    - bool: True if the scenario is connected, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(agent_comm_matrix)
    zero_eigenvalues_count = np.sum(np.abs(eigenvalues) < 1e-12)
    if flag_display:
        print(f"[verify_scenario_connectivity] Number of zero eigenvalues: {zero_eigenvalues_count}")
    if zero_eigenvalues_count > 1:
        if flag_display:
            print(f"[verify_scenario_connectivity] The graph has {zero_eigenvalues_count+1} connected components, not fully connected.")
            print("[verify_scenario_connectivity] WARNING - Regeneration of the scenario is required.")
        return False
    else:
        if flag_display:
            print("[verify_scenario_connectivity] The graph is connected. The scenario look okay")   
        return True

def generate_task_or_agent_location(num_entities, limit, existing_locations=None, gap=15, is_task=False, deployment_type='square'):
    """
    Generates locations for tasks or agents ensuring they respect specified constraints like minimum distance between them,
    and, for agents, their deployment shape.
    
    Parameters:
    - num_entities (int): Number of entities (tasks or agents) to generate.
    - limit (np.array): Array with the maximum range for location generation.
    - existing_locations (np.array): Already generated locations of the other type (for tasks, the existing agent locations).
    - gap (float): Minimum distance between entities.
    - is_task (bool): Flag indicating if generating locations for tasks.
    - deployment_type (str): Deployment pattern for agents ('circle', 'square', or 'skewed_circle').
    
    Returns:
    - np.array: Generated locations for the entities.
    """
    locations = np.zeros((num_entities, 2))
    for i in range(num_entities):
        ok = False
        while not ok:
            candidate = np.random.uniform(-limit, limit, 2)

            if i > 0 and np.any(np.linalg.norm(locations[:i] - candidate, axis=1) < gap):  # Check against other entities: their locations should not be too close to each other
                continue
            
            
            if is_task is not True: # Additional conditions for agents based on deployment type
                if deployment_type == 'square': 
                    pass
                if deployment_type == 'circle' and np.linalg.norm(candidate) >= np.min(limit):
                    continue
                elif deployment_type == 'skewed_circle' and np.abs(candidate[0] + candidate[1]) >= np.min(limit):
                    continue
                
            if is_task is True: # Additional conditions for tasks
                if np.any((limit*0.8 > np.abs(candidate))): # task position should not be too inside
                    continue 
                
            locations[i] = candidate
            ok = True
    return locations

def generate_scenario(num_tasks, num_agents, comm_distance=50, gap_agent=15, gap_task=200, task_location_range=[500, 500], agent_location_range=[300, 300], deployment_type='circle'):
    """
    Generates a scenario with specified parameters, creating task and agent locations.
    
    Parameters:
    - num_tasks (int): Number of tasks.
    - num_agents (int): Number of agents.
    - comm_distance, gap_agent, gap_task (int): Parameters specifying distances for communication, and minimum gaps between agents and tasks.
    - task_location_range, agent_location_range (list): The maximum range for task and agent locations.
    - deployment_type (str): The deployment pattern for agents ('circle', 'square', or 'skewed_circle').
    
    Returns:
    - dict: A dictionary containing the generated scenario data including agent and task locations, demands, and the communication matrix.
    """
    agent_locations = generate_task_or_agent_location(num_agents, np.array(agent_location_range), gap=gap_agent, deployment_type=deployment_type)
    print(f"[generate_scenario] Generated agent_locations")
    task_locations = generate_task_or_agent_location(num_tasks, np.array(task_location_range), gap=gap_task, is_task=True)
    print(f"[generate_scenario] Generated task_locations")
        
    agent_resources = np.random.uniform(10, 60, num_agents) # agent resource (e.g., energy, capacity)
    task_demands = np.random.uniform(1000 * num_agents / num_tasks, 2000 * num_agents / num_tasks, num_tasks)

    # Communication matrix generation
    dist_agents = np.linalg.norm(agent_locations[:, np.newaxis, :] - agent_locations[np.newaxis, :, :], axis=-1)
    agent_comm_matrix = (dist_agents <= comm_distance).astype(int) - np.eye(num_agents, dtype=int)

    environment = {
        'task_locations': task_locations, 
        'task_demands': task_demands, 
        'agent_locations': agent_locations,
        'agent_resources': agent_resources
        }
    
    initial_allocation = np.full(num_agents, -1, dtype=int)  # Initial allocation; "-1" indicates no task assigned

    print(f"[generate_scenario] Done")
    
    return {
        'initial_allocation': initial_allocation, 
        'agent_comm_matrix': agent_comm_matrix, 
        'num_agents': num_agents, 
        'num_tasks': num_tasks, 
        'environment': environment
        }

## Functions for Utility Function
def calculate_utility(agent_id, task_id, current_alloc, environment, util_type='logarithm_reward', ignore_cost=False):
    """
    Calculates the utility of an agent for a task, considering various utility types and whether to ignore the cost.
    
    Parameters:
    - agent_id (int): ID of the agent.
    - task_id (int): ID of the task.
    - current_alloc (list): the current allocation information over the tasks
    - num_participants (int): Number of participants allocated to the task.
    - environment (dict): Contains agent locations, task locations, and task demands.
    - util_type (str): Type of utility ('peaked_reward', 'logarithm_reward', 'constant_reward', 'energy_balanced', 'random').
    - ignore_cost (bool): If True, cost based on distance is not considered in utility calculation.
    
    Returns:
    - float: Calculated utility value.
    """
    agent_locations = environment['agent_locations']
    agent_resources = environment['agent_resources']
    task_locations = environment['task_locations']
    task_demands = environment['task_demands']
    
    # Extract the information about the task-specific coalition that the agent is currently belonging to
    current_members = (np.array(current_alloc) == task_id)
    current_members[agent_id] = True  # Include oneself
    num_participants = np.sum(current_members)    
    
    # Calculate cost as the Euclidean distance between agent and task, with option to ignore it
    cost = 0 if ignore_cost else np.linalg.norm(task_locations[task_id] - agent_locations[agent_id])
    
    if util_type == 'peaked_reward':
        # Adjusting for Peaked Reward calculation
        desired_num_agent_adjust_factor = 2 #
        num_agents = len(agent_locations)
        # Calculate the desired number of agents for this task based on its demand relative to total demand
        desired_num_agents = max(round(task_demands[task_id] / np.sum(task_demands) * num_agents * desired_num_agent_adjust_factor), 1)
        utility = task_demands[task_id] / desired_num_agents * np.exp(-num_participants / desired_num_agents + 1) - cost
    elif util_type == 'logarithm_reward':
        utility = task_demands[task_id] / (np.log2(len(agent_locations) / len(task_locations) + 1)) * np.log2(num_participants + 1) / num_participants - cost
    elif util_type == 'constant_reward':
        utility = task_demands[task_id] / num_participants - cost
    elif util_type == 'energy_balanced': # based on `constant_reward`
        current_members_resources = agent_resources[current_members]
        agent_contribution = agent_resources[agent_id]/np.sum(current_members_resources)
        utility = task_demands[task_id] * agent_contribution - cost        
    elif util_type == 'random':
        utility = np.random.rand()
    else:
        raise ValueError(f"Unsupported utility type: {util_type}")
    
    return max(utility, 0) # Ensure non-negative utility

## Functions for Main Algorithm: GRAPE
def distributed_mutex(agents, agent_comm_matrix):
    """
    Performs a distributed consensus (mutex) operation among agents based on their communication network.
    The decision is based on the highest iteration number among agents; in case of a tie, the latest timestamp decides.


    Parameters:
    - agents (list of dicts): Current local information of each agent, including allocation ('allocation'), iteration, timestamp, and satisfaction flag.
    - agent_comm_matrix (np.ndarray): Adjacency matrix representing the communication network between agents.

    Returns:
    - list of dicts: Updated state of each agent after consensus.
    """
    num_agents = len(agents)
    agent_updates = [agent.copy() for agent in agents]  # Create a copy for updates to avoid direct modification

    for agent_id in range(num_agents):
        # Identifying neighbouring agents based on the communication matrix, including the agent itself
        neighbour_ids = np.where(agent_comm_matrix[agent_id] > 0)[0]
        neighbour_ids_inclusive = np.append(neighbour_ids, agent_id)

        # Extract iterations and timestamps for the agent and its neighbours
        iterations = np.array([agents[id]['iteration'] for id in neighbour_ids_inclusive])
        timestamps = np.array([agents[id]['time_stamp'] for id in neighbour_ids_inclusive])

        # Determine the agent with the maximum iteration number
        max_iteration = np.max(iterations)
        agents_max_iter = neighbour_ids_inclusive[iterations == max_iteration]

        # Resolve ties with the latest timestamp
        if len(agents_max_iter) > 1:
            timestamps_max_iter = timestamps[iterations == max_iteration]
            decision_maker_id = agents_max_iter[np.argmax(timestamps_max_iter)]
        else:
            decision_maker_id = agents_max_iter[0]

        # Prepare the update based on the decision maker's data
        update_info = {
            'id': agent_id,
            'allocation': agents[decision_maker_id]['allocation'].copy(),
            'iteration': agents[decision_maker_id]['iteration'],
            'time_stamp': agents[decision_maker_id]['time_stamp'],
            'satisfied_flag': agent_updates[agent_id]['satisfied_flag']  # Preserve the original satisfaction flag for now
        }

        # Check for allocation changes to update the satisfaction flag accordingly
        if not np.array_equal(update_info['allocation'], agents[agent_id]['allocation']):
            update_info['satisfied_flag'] = False  # Mark as not satisfied if allocation changes
        else:
            update_info['satisfied_flag'] = True  # Otherwise, keep or mark as satisfied

        # Apply the prepared update
        agent_updates[agent_id].update(update_info)

    # Finalize updates by replacing the original agents' states with the updated ones
    for i in range(num_agents):
        agents[i].update(agent_updates[i])

    return agents

def grape_allocation(scenario, display_progress=True, util_type='constant_reward'):
    """
    Executes the GRAPE task allocation algorithm to assign agents to tasks based on their utilities.

    Parameters:
    - scenario (dict): Scenario data including agent and task information, and the communication network information.
    - display_progress (bool): Whether to print progress during allocation.

    Returns:
    - dict: Contains final allocations, utility, iteration data, and any allocation issues.
    """
    num_agents = scenario['num_agents']
    num_tasks = scenario['num_tasks']
    environment = scenario['environment']
    agent_comm_matrix = scenario['agent_comm_matrix']
    initial_allocation = scenario['initial_allocation']

    # Initialize agents' data structure. This is local information for each agent
    agents = [{
        'id': i,
        'allocation': initial_allocation.copy(),
        'iteration': 0,
        'time_stamp': np.random.rand(),
        'satisfied_flag': False,
        'util': 0.0 # Initialise as zero
    } for i in range(num_agents)]

    # For visualisation 
    consensus_step = 0
    satisfied_agents_count = 0
    allocation_history = []
    satisfied_agents_count_history = []
    iteration_history = []

    while satisfied_agents_count < num_agents: # until every agent is happy
        satisfied_agents_count = 0
        for agent_id in range(num_agents): # Each agent makes its own decision using its local information
            agent = agents[agent_id]
            current_alloc = agent['allocation']
            current_task = current_alloc[agent_id]
            candidates = np.full(num_tasks, -np.inf)

            for task_id in range(num_tasks):
                candidates[task_id] = calculate_utility(agent_id, task_id, current_alloc, environment, util_type=util_type)

            best_task = np.argmax(candidates)
            best_utility = candidates[best_task]
            if best_utility == 0:
                best_task = -1  # Go to the void; NOTE: void task is -1            

            if current_task == best_task:
                agent['satisfied_flag'] = True
                agent['util'] = best_utility
                satisfied_agents_count += 1
            else:
                agent['satisfied_flag'] = False
                agent['time_stamp'] = np.random.rand()                
                agent['iteration'] += 1
                agent['allocation'][agent_id] = best_task
                agent['util'] = 0.0 # As the agent needs to confirm again
                




        # Distributed Mutex 
        agents = distributed_mutex(agents, agent_comm_matrix)

        # Checking and updating satisfaction status
        if display_progress and consensus_step % 10 == 0:
            print(f"[grape_allocation] Iteration: {consensus_step}, Satisfied Agents: {satisfied_agents_count}/{num_agents}")

        # Final local information
        final_allocation = [agent['allocation'][agent_id] for agent_id in range(num_agents)]
        final_utilities = [agent['util'] for agent in agents]

        # Save data for each consensus step
        consensus_step += 1
        allocation_history.append(final_allocation)
        satisfied_agents_count_history.append(satisfied_agents_count)
        iteration_history.append(max(agent["iteration"] for agent in agents))

    print(f"[grape_allocation] Done")
                
    history = {
        'allocation': allocation_history,
        'satisfied_agents_count': satisfied_agents_count_history,
        'iteration': iteration_history
    }

    return {
        'final_allocation': final_allocation,
        'final_utilities': final_utilities,
        'consensus_step': consensus_step,
        'history': history,
        'problem_flag': satisfied_agents_count != num_agents
    }

## Visualisation 
def visualise_utility(agent_id, task_id, environment, max_participants, util_types=['peaked_reward', 'logarithm_reward', 'constant_reward', 'random'], filename='fig_utility_type'):
    """
    Visualises the change in utility value with the number of participants for various utility types.
    
    Parameters:
    - agent_id (int): ID of the agent.
    - task_id (int): ID of the task.
    - environment (dict): Contains agent locations, task locations, and task demands.
    - max_participants (int): Maximum number of participants to visualise.
    - util_types (list of str): Utility types to visualise.
    - filename (str): The filename where to save the visualization.
    """
    plt.figure(figsize=(10, 6))
    participant_range = np.arange(1, max_participants + 1)
    
    for util_type in util_types:
        utilities = []
        for num_participant in participant_range:
            current_alloc = [task_id if i < num_participant else -1 for i in range(num_agents)]
            utility = calculate_utility(agent_id, task_id, current_alloc, environment, util_type, ignore_cost=True)
            utilities.append(utility)
        plt.subplot(1, 2, 1)
        plt.plot(participant_range, utilities, label=util_type)
        plt.subplot(1, 2, 2)
        plt.plot(participant_range, participant_range * np.array(utilities), label=f'Sum {util_type} (no cost)')
    
    plt.subplot(1, 2, 1)
    plt.title('Individual Utility Value Changes')
    plt.xlabel('Number of Participants')
    plt.ylabel('Individual Utility Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title('Aggregated Utility Value Changes')
    plt.xlabel('Number of Participants')
    plt.ylabel('Aggregated Utility Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()

def visualise_scenario(scenario, final_allocation = None, filename="result_vis.png", agent_heterogeneity = False):
    """
    Visualises the final allocation of agents to tasks, including the connections between agents.
    
    Parameters:
    - scenario (dict): The scenario data including environment and the communication network information.
    - final_allocation (list): The final allocation of agents to tasks.
    - filename (str): The filename where to save the visualization.
    """
    task_locations = scenario['environment']['task_locations']
    agent_locations = scenario['environment']['agent_locations']
    task_demands = scenario['environment']['task_demands']
    agent_resources = scenario['environment']['agent_resources']
    agent_comm_matrix = scenario['agent_comm_matrix']
    
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))
    
    # Normalize t_demand for visualization scaling
    task_demands_sizes = (task_demands - task_demands.min()) / (task_demands.max() - task_demands.min()) * 300 + 50 # Example scaling
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_ylim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Plot agents's communication network  
    for i, j in zip(*np.where(np.triu(agent_comm_matrix) > 0)):
        ax.plot([agent_locations[i, 0], agent_locations[j, 0]], [agent_locations[i, 1], agent_locations[j, 1]], '-', color='gray', linewidth=0.5)
            
    # Plot agents (colouring as the same as the task colour)
    for i, location in enumerate(agent_locations):
        alloc = final_allocation[i] if final_allocation is not None else -1
        colour = 'black' if alloc == -1 else colours[int(alloc)]
        markersize = agent_resources[i]/np.max(agent_resources)*10 if agent_heterogeneity is True else 5
        ax.plot(agent_locations[i, 0], agent_locations[i, 1], 'o', markersize=markersize, markeredgecolor='gray', markerfacecolor=colour)
            

    
    # Plot tasks
    for i, location in enumerate(task_locations):
        ax.scatter(location[0], location[1], s=task_demands_sizes[i], c=[colours[i]], label=f'Task {i+1}', marker='s')
        
    title_name = 'Task Allocation Result' if final_allocation is not None else 'Locations of Agents and Tasks with Communication Links'
    plt.title(title_name) 
    plt.legend()
    if filename is None:
        pass
    else:
        plt.savefig(filename, dpi=300)
        plt.close()

def generate_gif_from_history(scenario, allocation_result, filename='result_animation.gif'):
    
        
    images = []  # To store paths of generated images
    
    print(f"[generate_gif_from_history] Generating temporary images for GIF")
    final_time = len(allocation_result['history']['iteration'])
    for k in range(0, final_time, 10):  # Step through history every 10 iterations
        current_filename = f"temp_{k}.png"
        current_alloc = allocation_result['history']['allocation'][k] 
        visualise_scenario(scenario, current_alloc, filename=current_filename)
        images.append(current_filename)
    
    # Create GIF
    with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
        for image_path in images:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    # Cleanup: Remove temporary image files 
    for image_path in images:
        os.remove(image_path)

    print(f"[generate_gif_from_history] Done")



if __name__ == "__main__":
    
    """
    A test function to demonstrate the generation of a scenario, task allocation, and final visualization.
    """
    
    # Scenario
    num_tasks = 3
    num_agents = 2**5*num_tasks
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