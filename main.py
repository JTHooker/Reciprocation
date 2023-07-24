from mesa import Agent

class SocialAgent(Agent):
    def __init__(self, unique_id, model, name):
        super().__init__(unique_id, model)
        self.name = name

    def greet(self, other_agent):
        print(f"{self.name} says: Hello, {other_agent.name}!")

from mesa import Model
from mesa.time import RandomActivation

class SocialInteractionModel(Model):
    def __init__(self):
        self.schedule = RandomActivation(self)
        agent1 = SocialAgent(1, self, "Alice")
        agent2 = SocialAgent(2, self, "Bob")
        self.schedule.add(agent1)
        self.schedule.add(agent2)

    def step(self):
        # For now, let's make Alice greet Bob every step.
        alice = self.schedule.agents[0]
        bob = self.schedule.agents[1]
        alice.greet(bob)

if __name__ == "__main__":
    model = SocialInteractionModel()
    model.step()
