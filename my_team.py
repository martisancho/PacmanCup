# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from util import Counter


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ImprovedOffensiveAgent', second='ImprovedDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents that evaluate states and actions.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.choke_points = []

    def register_initial_state(self, game_state):
        """
        Called at the start of the game to initialize agent's position and map info.
        """
        self.start = game_state.get_agent_position(self.index)
        self.choke_points = self.identify_choke_points(game_state)
        CaptureAgent.register_initial_state(self, game_state)

    def identify_choke_points(self, game_state):
        """
        Identifies choke points (narrow paths) in the map.
        """
        width, height = game_state.data.layout.width, game_state.data.layout.height
        choke_points = []

        # Analyze the map to find narrow corridors
        for x in range(width):
            for y in range(height):
                if game_state.has_wall(x, y):
                    continue
                neighbors = sum(
                    [not game_state.has_wall(nx, ny)
                     for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]]
                )
                if neighbors == 2:  # Potential choke point
                    choke_points.append((x, y))
        return choke_points

    def get_successor(self, game_state, action):
        """
        Finds the next state after applying the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns features for evaluation. To be implemented by subclasses.
        """
        return util.Counter()

    def get_weights(self, game_state, action):
        """
        Returns weights for evaluation. To be implemented by subclasses.
        """
        return {}

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # Evaluate all possible actions
        values = [self.evaluate(game_state, a) for a in actions]

        # Select the best actions
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Return a random choice among the best
        return random.choice(best_actions)


from util import manhattan_distance

class ImprovedOffensiveAgent(ReflexCaptureAgent):
    """
    Offensive agent that patrols the upper part of the map, attacks invaders,
    and returns to defend when necessary.
    """
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.center_position = None
        self.patrol_positions = []
        self.current_patrol_index = 0
        self.mode = "defend"
        self.has_collected_dot = False
        self.attack_start_time = None

    def register_initial_state(self, game_state):
        """
        Initialize patrol positions and the center position on our side.
        """
        super().register_initial_state(game_state)

        width = game_state.data.layout.width
        height = game_state.data.layout.height

        mid_x = (width // 2) - 1 if self.red else (width // 2)
        self.center_position = (mid_x, height // 2)

        self.patrol_positions = [
            (mid_x, self.center_position[1] - 5),
            (mid_x, self.center_position[1] - 3),
            (mid_x - 1, self.center_position[1] - 4),
            (mid_x + 1, self.center_position[1] - 4),
        ]

        self.patrol_positions = [
            pos for pos in self.patrol_positions
            if self.is_position_valid(pos, game_state)
        ]

    def choose_action(self, game_state):
        """
        Chooses an action based on the current mode.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if my_pos == self.start and self.mode == "attack":
            self.mode = "defend"

        if self.mode == "attack" and self.attack_start_time is not None:
            elapsed_time = game_state.data.timeleft - self.attack_start_time
            if elapsed_time >= 10000:  # 10 seconds = 1000 ms
                self.mode = "defend"

        if self.mode == "defend":
            return self.defensive_action(game_state, my_pos)
        else:
            return self.offensive_action(game_state, my_pos)

    def defensive_action(self, game_state, my_pos):
        """
        Handles defensive behavior.
        """
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            target = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            target_pos = target.get_position()
            best_action = min(
                actions,
                key=lambda a: self.get_maze_distance(
                    self.get_successor(game_state, a).get_agent_position(self.index),
                    target_pos,
                ),
            )
            if self.get_maze_distance(my_pos, target_pos) == 1:
                self.mode = "attack"
                self.has_collected_dot = False
                self.attack_start_time = game_state.data.timeleft
            return best_action
        else:
            patrol_target = self.patrol_positions[self.current_patrol_index]
            if my_pos == patrol_target:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_positions)

            return min(
                actions,
                key=lambda a: self.get_maze_distance(
                    self.get_successor(game_state, a).get_agent_position(self.index),
                    patrol_target,
                ),
            )

    def offensive_action(self, game_state, my_pos):
        """
        Handles offensive behavior with defender avoidance.
        """
        actions = game_state.get_legal_actions(self.index)
        food_list = self.get_food(game_state).as_list()

        if self.has_collected_dot:
            target = self.center_position
            if my_pos == target:
                self.mode = "defend"
            return self.avoid_defenders(game_state, actions, target)
        else:
            if len(food_list) > 0:
                target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
                best_action = self.avoid_defenders(game_state, actions, target)
                successor_pos = self.get_successor(game_state, best_action).get_agent_position(self.index)
                if successor_pos == target:
                    self.has_collected_dot = True
                return best_action
            else:
                self.mode = "defend"
                return random.choice(actions)

    def avoid_defenders(self, game_state, actions, target):
        """
        Avoid defenders while navigating toward the target.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        safe_actions = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)

            # Check safety of the action
            safe = True
            for defender in defenders:
                defender_pos = defender.get_position()
                if defender_pos and self.get_maze_distance(successor_pos, defender_pos) < 3:
                    safe = False
                    break
            if safe:
                safe_actions.append(action)

        # If no safe actions, fallback to all actions
        best_actions = safe_actions if safe_actions else actions

        # Move toward the target
        return min(
            best_actions,
            key=lambda a: self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_position(self.index),
                target,
            ),
        )

    def is_position_valid(self, pos, game_state):
        """
        Check if a position is valid and within our side of the map.
        """
        x, y = pos
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = (width // 2) - 1 if self.red else (width // 2)

        return 0 <= x < width and 0 <= y < height and not game_state.has_wall(x, y) and (x <= mid_x if self.red else x >= mid_x)




    def get_features(self, game_state, action):
        """
        Defines features to evaluate actions.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        # Defensive or offensive mode
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Distance to nearest invader
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, nearest_point(a.get_position())) for a in invaders]
            features['invader_distance'] = min(dists)

        # Penalize stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Weights for evaluating features.
        """
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
        }
    


class ImprovedDefensiveAgent(ReflexCaptureAgent):
    """
    Enhanced defensive agent:
    - Defends and patrols its side to catch attackers.
    - Briefly switches to offensive mode after eating an attacker.
    - Collects one dot while avoiding defenders and returns home.
    - Returns to defense mode after 10 seconds if no dot is collected.
    - Immediately switches back to defense mode if eaten.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.center_position = None
        self.patrol_positions = []
        self.current_patrol_index = 0
        self.mode = "defend"
        self.has_collected_dot = False
        self.attack_start_time = None

    def register_initial_state(self, game_state):
        """
        Initialize patrol positions and the center position on our side.
        """
        super().register_initial_state(game_state)

        width = game_state.data.layout.width
        height = game_state.data.layout.height

        mid_x = (width // 2) - 1 if self.red else (width // 2)
        self.center_position = (mid_x, height // 2)

        self.patrol_positions = [
            (mid_x, self.center_position[1] - 2),
            (mid_x, self.center_position[1] + 2),
            (mid_x - 1, self.center_position[1]),
            (mid_x + 1, self.center_position[1]),
        ]

        self.patrol_positions = [
            pos for pos in self.patrol_positions
            if self.is_position_valid(pos, game_state)
        ]

    def choose_action(self, game_state):
        """
        Chooses an action based on the current mode.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if my_pos == self.start and self.mode == "attack":
            self.mode = "defend"

        if self.mode == "attack" and self.attack_start_time is not None:
            elapsed_time = game_state.data.timeleft - self.attack_start_time
            if elapsed_time >= 10000:  # 10 seconds = 1000 ms
                self.mode = "defend"

        if self.mode == "defend":
            return self.defensive_action(game_state, my_pos)
        else:
            return self.offensive_action(game_state, my_pos)

    def defensive_action(self, game_state, my_pos):
        """
        Handles defensive behavior.
        """
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            target = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            target_pos = target.get_position()
            best_action = min(
                actions,
                key=lambda a: self.get_maze_distance(
                    self.get_successor(game_state, a).get_agent_position(self.index),
                    target_pos,
                ),
            )
            if self.get_maze_distance(my_pos, target_pos) == 1:
                self.mode = "attack"
                self.has_collected_dot = False
                self.attack_start_time = game_state.data.timeleft
            return best_action
        else:
            patrol_target = self.patrol_positions[self.current_patrol_index]
            if my_pos == patrol_target:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_positions)

            return min(
                actions,
                key=lambda a: self.get_maze_distance(
                    self.get_successor(game_state, a).get_agent_position(self.index),
                    patrol_target,
                ),
            )

    def offensive_action(self, game_state, my_pos):
        """
        Handles offensive behavior with defender avoidance.
        """
        actions = game_state.get_legal_actions(self.index)
        food_list = self.get_food(game_state).as_list()

        if self.has_collected_dot:
            target = self.center_position
            if my_pos == target:
                self.mode = "defend"
            return self.avoid_defenders(game_state, actions, target)
        else:
            if len(food_list) > 0:
                target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
                best_action = self.avoid_defenders(game_state, actions, target)
                successor_pos = self.get_successor(game_state, best_action).get_agent_position(self.index)
                if successor_pos == target:
                    self.has_collected_dot = True
                return best_action
            else:
                self.mode = "defend"
                return random.choice(actions)

    def avoid_defenders(self, game_state, actions, target):
        """
        Avoid defenders while navigating toward the target.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        safe_actions = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)

            # Check safety of the action
            safe = True
            for defender in defenders:
                defender_pos = defender.get_position()
                if defender_pos and self.get_maze_distance(successor_pos, defender_pos) < 3:
                    safe = False
                    break
            if safe:
                safe_actions.append(action)

        # If no safe actions, fallback to all actions
        best_actions = safe_actions if safe_actions else actions

        # Move toward the target
        return min(
            best_actions,
            key=lambda a: self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_position(self.index),
                target,
            ),
        )

    def is_position_valid(self, pos, game_state):
        """
        Check if a position is valid and within our side of the map.
        """
        x, y = pos
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = (width // 2) - 1 if self.red else (width // 2)

        return 0 <= x < width and 0 <= y < height and not game_state.has_wall(x, y) and (x <= mid_x if self.red else x >= mid_x)




    def get_features(self, game_state, action):
        """
        Defines features to evaluate actions.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        # Defensive or offensive mode
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Distance to nearest invader
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, nearest_point(a.get_position())) for a in invaders]
            features['invader_distance'] = min(dists)

        # Penalize stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Weights for evaluating features.
        """
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
        }








