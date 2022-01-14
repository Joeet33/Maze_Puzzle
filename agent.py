import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

from models.sarsa import SarsaTableModel

from game import Maze, Render

# indicates the episode, win/loss and time
logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    SARSA_ELIGIBILITY = auto()


maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])

game = Maze(maze)
# renders in model
game.render(Render.TRAINING)
model = SarsaTableModel(game)
h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                         stop_at_convergence=True)

# graphs showing win rate and cumulative rewards
try:
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

game.render(Render.MOVES)
game.play(model, start_cell=(4, 1))

plt.show()
