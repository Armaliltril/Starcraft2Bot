import random
import math
import os

import numpy as np
import pandas as pd
import constants as c

from sklearn.cluster import KMeans
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from counter import Counter


class QLearningTable:
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.5):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        state_action = self.q_table.ix[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        s_rewards = self.q_table.ix[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class Base_Protoss_Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Base_Protoss_Agent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(c.smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.nexus_y = None
        self.nexus_x = None

        self.move_number = 0

        self.counter = Counter()

        if os.path.isfile(c.DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(c.DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    # Only attack actions have '_'
    def splitAction(self, action_id):
        smart_action = c.smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def getFreePlaceForBuild(pylon_place):

        return None

    def getRandomUnit(self, unit_type, given_type):
        unit_y, unit_x = (unit_type == given_type).nonzero()

        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]

            return target

        return None

    def selectRandomUnit(self, unit_type, given_type):
        target = self.getRandomUnit(unit_type, given_type)
        if target is not None:
            return actions.FunctionCall(c._SELECT_POINT, [c._NOT_QUEUED, target])
        return None

    def selectAllUnits(self, unit_type, given_type):
        target = self.getRandomUnit(unit_type, given_type)
        if target is not None:
            return actions.FunctionCall(c._SELECT_POINT, [c._SELECT_ALL, target])
        return None

    def findClosestFreeVespene(self, unit_type, to_point):
        vespene_y, vespene_x = (unit_type == c._NEUTRAL_VESPENE_GEYSER).nonzero()
        vespene_geyser_count = int(math.ceil(len(vespene_y) / 97))

        geysers_coordinates = []
        for i in range(0, len(vespene_y)):
            geysers_coordinates.append((vespene_x[i], vespene_y[i]))

        kmeans = KMeans(n_clusters=vespene_geyser_count)
        kmeans.fit(geysers_coordinates)

        geysers = kmeans.cluster_centers_

        return self.getClosestPoint(to_point, geysers)

    def getClosestPoint(self, given_point, list_of_points):
        min_distance = 10e6
        index_of_closest = 0
        for i, point in enumerate(list_of_points):
            current_distance = self.distance(given_point, point)
            if current_distance < min_distance:
                min_distance = current_distance
                index_of_closest = i
        return list_of_points[index_of_closest]

    def distance(self, first_point, second_point):
        if len(first_point) == 2 and len(second_point) == 2:
            return abs(first_point[0] - second_point[0]) + abs(first_point[1] - second_point[1])

    def isAvailable(self, action, obs):
        if action in obs.observation['available_actions']:
            return True
        return False

    def step(self, obs):
        super(Base_Protoss_Agent, self).step(obs)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(c.DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            self.counter = Counter()

            return actions.FunctionCall(c._NO_OP, [])

        unit_type = obs.observation['screen'][c._UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][c._PLAYER_RELATIVE] == c._PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.nexus_y, self.nexus_x = (unit_type == c._PROTOSS_NEXUS).nonzero()

        nexus_y, nexus_x = (unit_type == c._PROTOSS_NEXUS).nonzero()
        self.counter.nexus = 1 if nexus_y.any() else 0

        # TODO: Rework for RL buildings
        pylon_y, pylon_x = (unit_type == c._PROTOSS_PYLON).nonzero()

        gateway_y, gateway_x = (unit_type == c._PROTOSS_GATEWAY).nonzero()

        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(12)
            current_state[0] = self.counter.nexus
            current_state[1] = self.counter.pylon
            current_state[2] = self.counter.gateway
            current_state[3] = obs.observation['player'][c._ARMY_]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['minimap'][c._PLAYER_RELATIVE] == c._PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            green_squares = np.zeros(4)
            friendly_y, friendly_x = (obs.observation['minimap'][c._PLAYER_RELATIVE] == c._PLAYER_SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))

                green_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            # actions we don't want to see
            excluded_actions = []

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action in c.build_actions:
                random_probe_selected = self.selectRandomUnit(unit_type, c._PROTOSS_PROBE)
                if random_probe_selected is not None:
                    return random_probe_selected

            elif smart_action in c.gateway_train_actions:
                gates_selected = self.selectAllUnits(unit_type, c._PROTOSS_GATEWAY)
                if gates_selected is not None:
                    return gates_selected

            elif smart_action in c.stargate_train_actions:
                stargates_selected = self.selectAllUnits(unit_type, c._PROTOSS_STARGATE)
                if stargates_selected is not None:
                    return stargates_selected

            elif smart_action in c.roboticsFaCility_train_actions:
                facilities_selected = self.selectAllUnits(unit_type, c._PROTOSS_ROBOTICS_FACILITY)
                if facilities_selected is not None:
                    return facilities_selected

            elif smart_action in c.nexus_train_actions:
                nexus_selected = self.selectAllUnits(unit_type, c._PROTOSS_NEXUS)
                if nexus_selected is not None:
                    return nexus_selected

            elif smart_action == c.ACTION_ATTACK:
                if c._SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(c._SELECT_ARMY, [c._NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            # TODO: make building better
            if smart_action == c.ACTION_BUILD_PYLON:
                if self.counter.pylon < 4 and self.isAvailable(c._BUILD_PYLON, obs):
                    if self.nexus_y.any():
                        x = round(self.nexus_x.mean())
                        y = round(self.nexus_y.mean())

                        if self.counter.pylon == 0:
                            target = self.transformDistance(x, 25, y, 0)
                        elif self.counter.pylon == 1:
                            target = self.transformDistance(x, 25, y, -15)
                        elif self.counter.pylon == 2:
                            target = self.transformDistance(x, 15, y, 5)
                        elif self.counter.pylon == 3:
                            target = self.transformDistance(x, 0, y, 15)

                        self.counter.pylon += 1
                        return actions.FunctionCall(c._BUILD_PYLON, [c._NOT_QUEUED, target])

            elif smart_action == c.ACTION_BUILD_GATEWAY:
                if self.counter.gateway < 3 and self.counter.pylon > 0 \
                        and self.isAvailable(c._BUILD_GATEWAY, obs):
                    if self.nexus_y.any():
                        x = round(self.nexus_x.mean())
                        y = round(self.nexus_y.mean())

                        if self.counter.gateway == 0:
                            target = self.transformDistance(x, 35, y, 0)
                        elif self.counter.gateway == 1:
                            target = self.transformDistance(x, 15, y, -10)
                        elif self.counter.gateway == 2:
                            target = self.transformDistance(x, 15, y, 25)

                        self.counter.gateway += 1
                        return actions.FunctionCall(c._BUILD_GATEWAY, [c._NOT_QUEUED, target])

            elif smart_action == c.ACTION_BUILD_CYBERNETICS_CORE:
                if self.counter.cyberneticsCore == 0 and self.counter.gateway > 0  \
                   and self.counter.pylon > 1 and self.isAvailable(c._BUILD_CYBERNETICS_CORE, obs):
                    if self.nexus_y.any():
                        x = round(self.nexus_x.mean())
                        y = round(self.nexus_y.mean())

                        target = self.transformDistance(x, 15, y, 15)

                        self.counter.cyberneticsCore += 1
                        return actions.FunctionCall(c._BUILD_CYBERNETICS_CORE, [c._NOT_QUEUED, target])

            elif smart_action == c.ACTION_BUILD_STARGATE:
                if self.counter.stargate == 0 and self.counter.cyberneticsCore > 0  \
                   and self.counter.pylon > 3 and self.isAvailable(c._BUILD_STARGATE, obs):
                    if self.nexus_y.any():
                        x = round(self.nexus_x.mean())
                        y = round(self.nexus_y.mean())

                        target = self.transformDistance(x, 30, y, 15)

                        self.counter.stargate += 1
                        return actions.FunctionCall(c._BUILD_STARGATE, [c._NOT_QUEUED, target])

            elif smart_action == c.ACTION_BUILD_DARK_SHRINE:
                if self.counter.darkShrine == 0 and self.counter.cyberneticsCore > 0  \
                   and self.counter.pylon > 3 and self.isAvailable(c._BUILD_DARK_SHRINE, obs):
                    if self.nexus_y.any():
                        x = round(self.nexus_x.mean())
                        y = round(self.nexus_y.mean())

                        target = self.transformDistance(x, 0, y, 25)

                        self.counter.stargate += 1
                        return actions.FunctionCall(c._BUILD_DARK_SHRINE, [c._NOT_QUEUED, target])

            elif smart_action == c.ACTION_BUILD_ASSIMILATOR:
                if self.counter.assimilator <= 2 and self.isAvailable(c._BUILD_ASSIMILATOR, obs):
                    if self.nexus_y.any():
                        vespene_y, vespene_x = (unit_type == c._NEUTRAL_VESPENE_GEYSER).nonzero()
                        i = random.randint(0, len(vespene_y) - 1)
                        x = vespene_x[i]
                        y = vespene_y[i]
                        target = [x, y]

                        self.counter.assimilator += 1
                        return actions.FunctionCall(c._BUILD_ASSIMILATOR, [c._NOT_QUEUED, target])

            elif smart_action in c.train_actions:
                function = c.action_to_function_map[smart_action]

                if smart_action is c.ACTION_TRAIN_PROBE and self.counter.probe <= 21:
                    if self.isAvailable(function, obs):
                        self.counter.probe += 1
                        self.counter.probe_minerals += 1
                        return actions.FunctionCall(function, [c._QUEUED])
                elif self.isAvailable(function, obs):
                    return actions.FunctionCall(function, [c._QUEUED])

            elif smart_action == c.ACTION_ATTACK:
                is_units_available_for_attack = True

                if len(obs.observation['single_select']) > 0 and \
                        obs.observation['single_select'][0][0] == c._PROTOSS_PROBE:
                    is_units_available_for_attack = False

                if len(obs.observation['multi_select']) > 0 and \
                        obs.observation['multi_select'][0][0] == c._PROTOSS_PROBE:
                    is_units_available_for_attack = False

                if is_units_available_for_attack and \
                        self.isAvailable(c._ATTACK_MINIMAP, obs):
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    location = self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))
                    return actions.FunctionCall(c._ATTACK_MINIMAP, [c._NOT_QUEUED, location])

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action in c.build_actions:
                if self.isAvailable(c._HARVEST_GATHER, obs):
                    randval = random.randint(1, 10)
                    if randval >= 3:
                        target = self.getRandomUnit(unit_type, c._NEUTRAL_MINERAL_FIELD)
                    else:
                        target = self.getRandomUnit(unit_type, c._PROTOSS_ASSIMILATOR)
                    if target is not None:
                        return actions.FunctionCall(c._HARVEST_GATHER, [c._QUEUED, target])

        return actions.FunctionCall(c._NO_OP, [])
