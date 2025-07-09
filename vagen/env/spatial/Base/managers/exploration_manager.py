import numpy as np
import copy
from typing import List, Tuple, Dict, Any

from ..core.object import Object, Agent
from ..core.relationship import DirPair, DirectionSystem, Dir
from ..core.graph import DirectionalGraph
from ..actions import (
    BaseAction,
    ActionSequence,
    MoveAction,
    RotateAction,
    ReturnAction,
    ObserveAction,
    TermAction,
)
from ..core.room import Room

"""
TODO:
1. exploration efficiency for Observe()
"""


class ExplorationManager:
    """Manages spatial exploration with agent movement and queries.
    
    Maintains base_room (original state) and exploration_room (with agent_anchor).
    Actions handle their own coordinate transformations. Exploration is egocentric.
    """
    
    def __init__(self, room: Room):
        assert room.agent is not None, "Exploration requires an agent in the room"
        
        self.base_room = room.copy()
        
        agent_anchor = Object("agent_anchor", room.agent.pos.copy(), room.agent.ori.copy())
        
        # Create exploration room with all objects including agent_anchor
        exploration_objects = copy.deepcopy(room.objects) + [agent_anchor]
        self.exploration_room = Room(
            objects=exploration_objects,
            name=f"{room.name}_exploration",
            agent=copy.deepcopy(room.agent)
        )
        
        # Get reference to agent_anchor from exploration room
        self.agent_anchor = self.exploration_room.get_object_by_name("agent_anchor")
        self.agent_idx = self._get_index(self.exploration_room.agent.name)
        self.anchor_idx = self._get_index(self.agent_anchor.name)

        self.objects = self.exploration_room.all_objects
        self.exp_graph = DirectionalGraph(self.objects, is_explore=True)
        
        self.exp_graph.add_edge(self.agent_idx, self.anchor_idx, DirPair(Dir.SAME, Dir.SAME))


        # log exploration efficiency
        self.n_valid_queries = 0
        self.n_novel_queries = 0
        
    def _get_index(self, name: str) -> int:
        """Get object index by name."""
        for i, obj in enumerate(self.exploration_room.all_objects):
            if obj.name == name:
                return i
        raise ValueError(f"Object '{name}' not found")
    
    def _update_move(self, target_name: str):
        """Update exploration graph after move action."""
        target_idx = self._get_index(target_name)
        self.exp_graph.move_node(self.agent_idx, target_idx, DirPair(Dir.SAME, Dir.SAME))

    def _update_rotate(self, degrees: int):
        """Update exploration graph after rotate action."""
        self.exp_graph.rotate_axis(degrees)
    
    def _update_observe(self, visible_objects: List[str]) -> bool:
        """Update exploration graph after observe action. TODO"""
        agent_idx = self._get_index("agent")
        agent = self.exploration_room.agent
        
        any_novel = False
        for obj_name in visible_objects:
            target_idx = self._get_index(obj_name)
            target_obj = self.objects[target_idx]
            
            dir_pair = DirectionSystem.get_direction(target_obj.pos, agent.pos, agent.ori)
            if self.exp_graph.add_edge(target_idx, agent_idx, dir_pair):
                any_novel = True
                
        return any_novel
    
    def _execute_and_update(self, action: BaseAction) -> Tuple[bool, str, Dict[str, Any]]:
        """Execute action and update exploration state."""
        if isinstance(action, ReturnAction):
            kwargs = {'agent_anchor': self.agent_anchor}
        elif isinstance(action, ObserveAction):
            kwargs = {'neglect_objects': [self.agent_anchor.name]}
        else:
            kwargs = {}
        result = action.execute(self.exploration_room, **kwargs)
        
        if not result.success:
            return result.success, result.message, result.data
        
        # success execution
        if isinstance(action, MoveAction):
            self._update_move(result.data['target_name'])
        elif isinstance(action, RotateAction):
            self._update_rotate(result.data['degrees'])
        elif isinstance(action, ReturnAction):
            self._update_move(result.data['target_name'])
            self._update_rotate(result.data['degrees'])
        elif isinstance(action, ObserveAction):
            result.data['novel_query'] = self._update_observe(result.data['visible_objects'])
        
        return result.success, result.message, result.data

    def execute_action(self, action: BaseAction) -> None:
        """Execute single action with validation."""
        success, message, _ = self._execute_and_update(action)
        if not success:
            raise ValueError(f"Action execution failed: {message}")
    
    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[str, Dict[str, Any]]:
        """Execute action sequence with validation."""
        if not action_sequence.final_action:
            return "Action sequence requires a final action.", {}
        
        if isinstance(action_sequence.final_action, TermAction) and action_sequence.motion_actions:
            return "Term() action should not have motion actions.", {}
        
        info = {'novel_query': False}
        messages = []
        
        for action in action_sequence.motion_actions:
            success, msg, action_info = self._execute_and_update(action)
            info.update(action_info)
            messages.append(msg)
            if not success:
                return ", ".join(messages), info
        
        success, msg, action_info = self._execute_and_update(action_sequence.final_action)
        info.update(action_info)
        messages.append(msg)
        if not success:
            return ", ".join(messages), info
        
        if info['novel_query']:
            self.n_novel_queries += 1
        if not action_sequence.final_action.is_term():
            self.n_valid_queries += 1
        
        return ", ".join(messages), info
    
    def finish_exploration(self, return_to_origin: bool = True, neglect_anchor: bool = True) -> Room:
        """Complete exploration and return final room state."""
        if return_to_origin and self.agent_anchor:
            result = self.execute_action(ReturnAction())
            assert result.success, "Return action failed"
            self._update_move(result.data['target_name'])
            self._update_rotate(result.data['degrees'])
            
        if neglect_anchor:
            self._remove_anchor()
        return self.exploration_room
    
    def _remove_anchor(self) -> None:
        """Remove anchor from exploration room and graph."""
        if not self.agent_anchor:
            return
            
        anchor_idx = self._get_index("agent_anchor")
        
        # Remove from exploration room
        self.exploration_room.objects = [obj for obj in self.exploration_room.objects if obj.name != "agent_anchor"]
        self.exploration_room.all_objects = [self.exploration_room.agent] + self.exploration_room.objects
        
        # Update exploration graph
        self.exp_graph.size -= 1
        for matrix_name in ['_v_matrix', '_h_matrix', '_v_matrix_working', '_h_matrix_working', '_asked_matrix']:
            matrix = getattr(self.exp_graph, matrix_name)
            matrix = np.delete(matrix, anchor_idx, axis=0)
            matrix = np.delete(matrix, anchor_idx, axis=1)
            setattr(self.exp_graph, matrix_name, matrix)
        
        self.objects = self.exploration_room.all_objects
        self.agent_anchor = None
    
    def get_unknown_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with unknown relationships."""
        return self.exp_graph.get_unknown_pairs()
    
    def get_inferable_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with inferable relationships."""
        return self.exp_graph.get_inferable_pairs()
    
    def get_exploration_efficiency(self) -> Dict[str, float]:
        """Get exploration efficiency metrics."""
        unknown_pairs = self.get_unknown_pairs()
        n_object = len(self.objects)
        max_rels = int(n_object * (n_object - 1) / 2)
        coverage = (max_rels - len(unknown_pairs)) / max_rels if max_rels > 0 else 0
            
        return {
            "coverage": coverage,
            "novelty": self.n_novel_queries / self.n_valid_queries if self.n_valid_queries > 0 else 0,
            "n_valid_queries": self.n_valid_queries,
            "n_novel_queries": self.n_novel_queries,
        }


if __name__ == "__main__":
    pass