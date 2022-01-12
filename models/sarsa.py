import logging
import random
from datetime import datetime

import numpy as np

from game import Status
from models.abstractmodel import AbstractModel


class SarsaTableModel(AbstractModel):
    default_check_convergence_every = 5

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        # table with Q per (state, action) combination
        self.Q = dict()

    # train the model
    def train(self, stop_at_convergence=False, **kwargs):
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        # percentage reduction per step = 100 - exploration decay
        exploration_decay = kwargs.get("exploration_decay", 0.995)
        learning_rate = kwargs.get("learning_rate", 0.10)
        # 0.80 = 20% reduction
        eligibility_decay = kwargs.get("eligibility_decay", 0.80)
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # variables for performance reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        # starts training
        for episode in range(1, episodes + 1):
            # make sure to start training from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())

            etrace = dict()

            if np.random.random() < exploration_rate:
                action = random.choice(self.environment.actions)
            else:
                action = self.predict(state)

            while True:
                try:
                    etrace[(state, action)] += 1
                except KeyError:
                    etrace[(state, action)] = 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                next_action = self.predict(next_state)

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                next_Q = self.Q.get((next_state, next_action), 0.0)

                delta = reward + discount * next_Q - self.Q[(state, action)]

                for key in etrace.keys():
                    self.Q[key] += learning_rate * delta * etrace[key]

                # decay the eligibility trace
                for key in etrace.keys():
                    etrace[key] *= (discount * eligibility_decay)
                # terminal state reached, stop training episode
                if status in (Status.WIN, Status.LOSE):
                    break

                state = next_state
                # SARSA is on-policy: always follow the predicted action
                action = next_action

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break
            # explore less as training progresses
            exploration_rate *= exploration_decay

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    # get q values for all actions for a certain state
    def q(self, state):
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    # selects the action with the highest value from the Q-table
    def predict(self, state):
        q = self.q(state)

        logging.debug("q[] = {}".format(q))
        # get index of the action(s) with the max value
        actions = np.nonzero(q == np.max(q))[0]
        return random.choice(actions)
