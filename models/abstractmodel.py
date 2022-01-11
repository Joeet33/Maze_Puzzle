from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")

    # loads model from file
    def load(self, filename):
        pass

    # saves model to file
    def save(self, filename):
        pass

    # trains model
    def train(self, stop_at_convergence=False, **kwargs):
        pass

    # returns q values for state
    @abstractmethod
    def q(self, state):
        pass

    # predicts value based on state
    @abstractmethod
    def predict(self, state):
        pass
