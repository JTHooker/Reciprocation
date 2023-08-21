from mesa import Agent, Model #imports modules from mesa
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.stats import beta
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer


alpha = 15  # Shape parameter for distribution of flexibility using gamma distribution
beta_param = 3.75  # Shape parameter for distribution of flexibility using gamma distribution

# Client Agent This sets up the Client agent and assigns them with features
class Client(Agent): #This line declares the Client class, which inherits from the Agent class provided by the Mesa library. By inheriting from Agent, the Client class gains access to all the properties and methods defined in the Agent class, allowing it to function as an agent within the Mesa framework.
    def __init__(self, unique_id, model): #This line defines the initialization method (__init__) of the Client class. The method takes three parameters: self, unique_id, and model. The self parameter refers to the instance of the Client class being created, unique_id is a unique identifier for the agent, and model is a reference to the model that the agent is part of.
        super().__init__(unique_id, model) #This line runs the line above from the parent class of Clint, which is 'Agent'.
        self.physical_health = 30 + 7 * self.random.normalvariate(0, 1) # everyone starts off with pretty bad health
        self.mental_health = 30 + 7 * self.random.normalvariate(0, 1)
        self.initial_physical_health = self.physical_health # sets initial physical health
        self.initial_mental_health = self.mental_health  # sets initial mental health
        self.flexibility = beta.rvs(alpha, beta_param)  # Draw a sample from the beta distribution
        self.service_request = (self.random.choice(["Green", "Orange", "Red"]), self.random.randint(0, 100)) #sets up a service request from the client that is later sent to the manager agent
        self.current_state = 1 #this is the state that the client first finds themselves in
        self.state_history = [self.current_state] #this is updated over time and then each state is appended in a list in the 'move to next state' function
        self.satisfaction = beta.rvs(alpha, beta_param) * 100  # Initial satisfaction level
        self.request_attempts = 0  # Counter for request attempts

        #Testing for changes

    def step(self):
        self.client_request()  # Client makes a request to the case manager, below, at each step.

    def client_request(self):
        request = self.service_request
        while True:
            # Finding the assigned case manager
            assigned_case_manager = [agent for agent in self.model.schedule.agents
                                     if isinstance(agent, CaseManager)
                                     and self.unique_id in agent.assigned_clients][0]
            response = assigned_case_manager.evaluate_request(request)
            if response == request:  # If the request is accepted
                self.move_to_next_state()
                self.request_attempts = 0  # Reset attempts counter for next request
                self.satisfaction = min(100, max(0, self.satisfaction * 1.05))  # Increase satisfaction
                break
            elif self.random.random() < self.flexibility:  # If the client accepts the modification
                self.service_request = response
                request = response  # Update request for the next iteration
                self.request_attempts += 1
                self.satisfaction = min(100, max(0, self.satisfaction * 0.98))  # Slightly decrease satisfaction
            else:  # If the client does not accept the modification
                self.request_attempts += 1
                self.satisfaction = min(100, max(0, self.satisfaction * 0.8))  # Significantly decrease satisfaction
                # Record the denied request in the system
                self.model.denied_requests += 1
                break
    def move_to_next_state(self):
        self.current_state += 1  # Increment state

        # Calculate physical health improvement
        max_physical_health = 100 - (self.initial_physical_health * 0.5)
        k = 0.002
        improvement_rate = 1 + (1 * np.exp(-k * self.physical_health))  # improvement per state change
        phys_improvement = self.physical_health * improvement_rate
        self.physical_health = min(self.physical_health + phys_improvement, max_physical_health)

        # Calculate mental health improvement
        max_mental_health = 100 - (self.initial_mental_health * 0.5)  # Define upper limit for mental health
        men_improvement = self.mental_health * improvement_rate
        self.mental_health = min(self.mental_health + men_improvement, max_mental_health)  # Update mental health with the limit

        # Check for absorbing state
        phys_absorbing_threshold = 90  # You can set this value to the desired threshold
        psych_absorbing_threshold = 90
        if self.physical_health >= phys_absorbing_threshold and self.mental_health >= psych_absorbing_threshold:
            self.current_state = 100
        self.state_history.append(self.current_state)
        # Record transition
        if len(self.state_history) > 1:
            previous_state = self.state_history[-2]
            self.model.transitions[(previous_state, self.current_state)] += 1


# Case Manager Agent
class CaseManager(Agent):
    def __init__(self, unique_id, model, assigned_clients):
        super().__init__(unique_id, model)
        self.guidelines = ["Green", "Orange"]
        self.restrictions = self.random.randint(0, 100)
        self.assigned_clients = assigned_clients
        self.current_client_index = 0

    def step(self):
        client = [agent for agent in self.model.schedule.agents if agent.unique_id == self.assigned_clients[self.current_client_index]][0]
        client.client_request()
        self.current_client_index = (self.current_client_index + 1) % len(self.assigned_clients)

    def evaluate_request(self, request): #manager evaluates the request from the client
        color, intensity = request #the request is made up of the category and the intensity of the request
        if color in self.guidelines and intensity <= self.restrictions:
            return request #if all good, return 'request'
        else:
            return (color, max(1, intensity * 0.9))

# Patient Representative Agent
class PatientRepresentative(Agent):
    def __init__(self, unique_id, model, client_id):
        super().__init__(unique_id, model)
        self.client_id = client_id

    def step(self):
        client = [agent for agent in self.model.schedule.agents if agent.unique_id == self.client_id][0]
        if client.mental_health < 50:
            client.client_request()

# Health Care Model
class HealthCareModel(Model):
    def __init__(self, N_clients=1000, N_case_managers=10, clients_per_manager=None, grid_size=(10, 10)):
        # Create the grid
        self.grid = MultiGrid(*grid_size, torus=False)
        self.transitions = defaultdict(int)
        self.num_clients = N_clients
        if clients_per_manager is None:
            clients_per_manager = N_clients // N_case_managers
        self.clients_per_manager = clients_per_manager
        self.num_clients = N_clients
        self.denied_requests = 0  # Counter for denied requests across the system
        self.num_case_managers = N_case_managers
        self.clients_per_manager = clients_per_manager
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            {"Average_Physical_Health": lambda m: sum([agent.physical_health for agent in m.schedule.agents if isinstance(agent, Client)]) / m.num_clients,
             "Average_Mental_Health": lambda m: sum([agent.mental_health for agent in m.schedule.agents if isinstance(agent, Client)]) / m.num_clients,
             "Clients_in_Absorbing_State": lambda m: len([agent for agent in m.schedule.agents if isinstance(agent, Client) and agent.current_state == 100]),
             "Denied_Requests": lambda m: m.denied_requests,
             "Average_Satisfaction": lambda m: sum([agent.satisfaction for agent in m.schedule.agents if isinstance(agent, Client)]) / m.num_clients,
            }
        )
        # Creating clients
        for i in range(self.num_clients):
            x = i % grid_size[0]
            y = (i // grid_size[0]) % grid_size[1]
            client = Client(i, self)
            self.grid.place_agent(client, (x, y))
            self.schedule.add(client)

        # Creating case managers
        # Creating case managers
        for i in range(self.num_clients, self.num_clients + self.num_case_managers):
            assigned_clients = list(range((i - self.num_clients) * self.clients_per_manager,
                                          (i - self.num_clients + 1) * self.clients_per_manager))
            case_manager = CaseManager(i, self, assigned_clients)
            x = self.random.randint(0, grid_size[0] - 1)  # Random x-coordinate
            y = self.random.randint(0, grid_size[1] - 1)  # Random y-coordinate
            self.grid.place_agent(case_manager, (x, y))
            self.schedule.add(case_manager) # Adds them to the scheduler. Agents don't have to be added to the schedule - this is flexible. They can exist outside it or be removed from it - kind of 'hidden' if you want.
        # Creating patient representatives
        for i in range(self.num_clients + self.num_case_managers,
                       self.num_clients + self.num_case_managers + self.num_clients//10):
            client_id = (i - self.num_clients - self.num_case_managers) * 10
            representative = PatientRepresentative(i, self, client_id)
            self.schedule.add(representative) #This process continues for 100 iterations, creating 100 patient representatives and linking them to clients with IDs from 0 to 990 (in increments of 10).

    def print_assigned_clients(self, case_manager_id):
        case_manager = [agent for agent in self.schedule.agents if
                        isinstance(agent, CaseManager) and agent.unique_id == case_manager_id][0]
        print(f"Case Manager ID: {case_manager.unique_id}")
        print(f"Assigned Clients: {case_manager.assigned_clients}")

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Simulation and Visualization Functions
def run_simulation(model, n_steps=1000):
    for i in range(n_steps):
        model.step()
    return model.datacollector.get_model_vars_dataframe()

def plot_average_physical_health(results):
    plt.figure(figsize=(10,5))
    plt.plot(results['Average_Physical_Health'])
    plt.title('Average Physical Health Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Average Physical Health')
    plt.show()

def plot_average_mental_health(results):
    plt.figure(figsize=(10,5))
    plt.plot(results['Average_Mental_Health'])
    plt.title('Average Mental Health Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Average Mental Health')
    plt.show()

def plot_clients_in_absorbing_state(results):
    plt.figure(figsize=(10,5))
    plt.plot(results['Clients_in_Absorbing_State'])
    plt.title('Clients in Absorbing State Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Number of Clients')
    plt.show()

def plot_denials(results):
    plt.figure(figsize=(10,5))
    plt.plot(results['Denied_Requests'])
    plt.title('Denied requests over time')
    plt.xlabel('Steps')
    plt.ylabel('Number of Denied Requests')
    plt.show()

def plot_satisfaction(results):
    plt.figure(figsize=(10,5))
    plt.plot(results['Average_Satisfaction'])
    plt.title('Mean Satisfaction over time')
    plt.xlabel('Steps')
    plt.ylabel('Level of Satisfaction')
    plt.show()

def visualize_transitions(model):
    G = nx.DiGraph()
    for (from_state, to_state), count in model.transitions.items():
        G.add_edge(from_state, to_state, weight=count)

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title('Transitions Between States')
    plt.show()

# Run and visualize
health_care_model = HealthCareModel()
results = run_simulation(health_care_model, 100)
plot_average_physical_health(results)
plot_average_mental_health(results)
plot_clients_in_absorbing_state(results)
plot_denials(results)
plot_satisfaction(results)
visualize_transitions(health_care_model)


health_care_model.print_assigned_clients(case_manager_id=1005) # Replace with the desired ID

#test

def agent_portrayal(agent):
    # Define how to draw the agent on the grid
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "r": 0.5
    }

    if isinstance(agent, Client):
        portrayal["Color"] = "red"
        if agent.current_state == 100:
            portrayal["Color"] = "green"
    elif isinstance(agent, CaseManager):
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "blue"
        portrayal["w"] = 0.1
        portrayal["h"] = 0.1
    elif isinstance(agent, PatientRepresentative):
        portrayal["Shape"] = "diamond"
        portrayal["Color"] = "purple"
        portrayal["r"] = 0.1

    return portrayal


average_physical_health_chart = ChartModule([{"Label": "Average_Physical_Health", "Color": "Black"}], data_collector_name='datacollector')

grid_size = (25, 40)
grid = CanvasGrid(agent_portrayal, *grid_size, 500, 500)
server = ModularServer(HealthCareModel, [grid, average_physical_health_chart], "Health Care Model", {"N_clients": 1000, "N_case_managers": 10, "grid_size": grid_size})

server.port = 8522
server.launch()
