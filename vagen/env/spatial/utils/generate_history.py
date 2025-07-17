"""
Generate exploration history using DFS
"""
import numpy as np
from collections import Counter

from typing import List, Tuple, Dict
from vagen.env.spatial.ToS_Base.tos_base import (
    Room,
    DirectionalGraph,
    DirPair,
    Dir,
    DirectionSystem,
    BaseAction,
    MoveAction,
    RotateAction,
    QueryAction,
    TermAction,
    ExplorationManager
)

"""
TODO re-implement this
"""

class AutoExplore:
    """
    Automatically explore the environment
    TODO: use BFS to get shortest path
    """
    
    def __init__(self, room: Room, np_random: np.random.Generator):
        self.room = room.copy()
        self.np_random = np_random
        self.exp_manager = ExplorationManager(self.room)

    

    def _generate_history_passive(self) -> List[Tuple]:
        """
        Generate exploration history of egocentric exploration using ExplorationManager
        
        Returns:
            history: list of ((obj1, obj2), dir_pair)
            actions: list of Action instances in chronological order
        """
        assert self.room.agent is not None, "Agent is not in the room"

        query_result, actions, actions_in_a_turn = [], [], []
        agent_idx = self.exp_manager._get_index(self.room.agent.name)

        while True:
            unknown_pairs = self.exp_manager.get_unknown_pairs()
            if not unknown_pairs:
                actions.append([TermAction()])
                break

            # Get unknowns involving agent (index 0)
            local_unknowns = [(i, j) if i == agent_idx else (j, i) for (i, j) in unknown_pairs if i == agent_idx or j == agent_idx]
            
            if not local_unknowns:

                # find the next object with most unknowns
                counts = Counter()
                for i, j in unknown_pairs:
                    counts[i] += 1
                    counts[j] += 1
                next_obj_idx = max(counts, key=counts.get)
                obj_name = self.exp_manager._objects[next_obj_idx].name
                
                # Rotate towards target if not visible
                agent = self.exp_manager._objects[agent_idx]
                target = self.exp_manager._objects[next_obj_idx]
                if not self.exp_manager._is_visible(agent, target):
                    # Find first rotation that makes target visible
                    rotation = next((r for r in [0, 90, 180, 270] 
                                  if self._would_be_visible_after_rotation(next_obj_idx, r)), 0)
                    if rotation:
                        action = RotateAction(rotation)
                        actions_in_a_turn.append(action)
                        self.exp_manager.execute_action(action)
                action = MoveAction(obj_name)
                actions_in_a_turn.append(action)
                self.exp_manager.execute_action(action)
                continue
            
            # Query all unknowns at current position
            while local_unknowns:

                # Check for visible unknowns in current direction first
                visible_unknowns = [target_idx for _, target_idx in local_unknowns 
                                  if self.exp_manager._is_visible(self.exp_manager._objects[agent_idx], 
                                                                self.exp_manager._objects[target_idx])]
                
                if not visible_unknowns:
                    # No visible unknowns in current direction, find best direction
                    best_direction = self._find_best_direction(local_unknowns)
                    if best_direction != 0:
                        action = RotateAction(best_direction)
                        actions_in_a_turn.append(action)
                        self.exp_manager.execute_action(action)
                        continue
                
                # Query all visible unknowns
                for target_idx in visible_unknowns:
                    obj_name = self.exp_manager._objects[target_idx].name
                    action = QueryAction(obj_name)
                    actions_in_a_turn.append(action)
                    self.exp_manager.execute_action(action)
                    
                    # Extract direction from manager's internal state
                    dir_pair = self.exp_manager.current_room.get_direction(
                        self.exp_manager._objects[target_idx].name, 
                        self.exp_manager._objects[agent_idx].name
                    )[0]
                    query_result.append((self.exp_manager._objects[target_idx].name, dir_pair))

                    # query marks the end of a turn
                    actions.append(actions_in_a_turn)
                    actions_in_a_turn = []
                
                # Update unknowns
                unknown_pairs = self.exp_manager.get_unknown_pairs()
                local_unknowns = [(i, j) if i == agent_idx else (j, i) for (i, j) in unknown_pairs if i == agent_idx or j == agent_idx]

        return query_result, actions

    def _find_best_direction(self, local_unknowns: List[Tuple[int, int]]) -> int:
        """Find direction with most visible unknowns"""
        best_count, best_direction = 0, 0
        
        for rotation in [0, 90, 180, 270]:
            count = sum(1 for _, target_idx in local_unknowns 
                       if self._would_be_visible_after_rotation(target_idx, rotation))
            if count > best_count:
                best_count, best_direction = count, rotation
        
        return best_direction

    def _would_be_visible_after_rotation(self, target_idx: int, rotation: int) -> bool:
        """Check visibility after rotation"""
            
        agent = self.exp_manager.current_room.agent
        target = self.exp_manager._objects[target_idx]
        if target.name == self.exp_manager.current_room.agent.name:
            return True
        
        rotations = {
            0: np.array([[1, 0], [0, 1]]),
            90: np.array([[0, -1], [1, 0]]),
            270: np.array([[0, 1], [-1, 0]]),
            180: np.array([[-1, 0], [0, -1]]),
        }
        rotated_ori = agent.ori @ rotations[rotation]
        temp_agent = type(target)(name='', pos=agent.pos, ori=rotated_ori)  
        return self.exp_manager._is_visible(temp_agent, target)
    
    def _format_history_to_string(self, query_result: List[Tuple], actions: List[List[BaseAction]]) -> str:
        """
        Convert history and actions from _generate_history_passive to formatted string.
        
        Args:
            query_result: List of (obj_name, dir_pair) tuples from _generate_history_passive
            actions: List of action lists, where each inner list represents one turn            
        Returns:
            Formatted string with numbered turns
        """
        turn_strings = []
        query_idx = 0  # Track current position in query_result
        
        for turn_num, turn_actions in enumerate(actions, 1):
            action_strings = []
            
            for action in turn_actions:
                if isinstance(action, QueryAction):
                    obj_name, dir_pair = query_result[query_idx]
                    dir_string = DirectionSystem.to_string(dir_pair, perspective="ego")
                    answer = f"{obj_name} is {dir_string}"
                    action_string = action.success_message(answer=answer)
                    query_idx += 1
                else:
                    # For non-query actions, no additional parameters needed
                    action_string = action.success_message()
                
                action_strings.append(action_string)
            
            # Concatenate all actions in this turn
            turn_string = " ".join(action_strings)
            turn_strings.append(f"{turn_num}. {turn_string}")
        
        return "\n".join(turn_strings)
    
    def gen_exp_history(self) -> str:
        return "test"
        query_result, actions = self._generate_history_passive()
        return self._format_history_to_string(query_result, actions)

if __name__ == "__main__":
    import re
    from vagen.env.spatial.Base import Object, Agent, generate_room, CANDIDATE_OBJECTS
    from gymnasium.utils import seeding

    rng1 = seeding.np_random(1024)[0]
    room = generate_room(
        room_range=(-10, 10),
        n_objects=3,
        candidate_objects=CANDIDATE_OBJECTS,
        generation_type='rand',
        perspective='ego',
        np_random=rng1,
    )
    print(room)

    explorer = AutoExplore(room, rng1)
    exploration_str = explorer.gen_exp_history()
    print(exploration_str)
