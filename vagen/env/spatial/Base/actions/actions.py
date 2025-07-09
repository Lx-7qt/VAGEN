from typing import Optional, List
import re
import numpy as np

from .base import BaseAction, ActionResult

"""
Specific action implementations for spatial exploration.
Contains all concrete action classes and the ActionSequence parser.
"""


class MoveAction(BaseAction):
    """Move to a target object"""
    
    format_desc = "Move(object_name)"
    description = "Move to a specific object in the room"
    example = "Move(table)"
    
    def __init__(self, target=None):
        super().__init__(target)
        self.target = target

    def move_agent_to_pos(self, room, target_pos):
        """Move agent to target position, shift coordinate system to keep agent at origin"""
        room.agent.pos = target_pos.copy()
        for obj in room.objects:
            if obj.name != room.agent.name:
                obj.pos = obj.pos - room.agent.pos
    
    def format_pattern(self) -> str:
        return r"^Move\(([A-Za-z0-9_-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Moved to {self.target}."
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "not visible"}
        return f"Cannot move to '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute move action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"))
        
        target_obj = room.get_object_by_name(self.target)
        if not self._is_visible(room.agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"))
        
        self.move_agent_to_pos(room, target_obj.pos)

        return ActionResult(True, self.get_feedback(True), {'target_name': self.target})
    
    def __repr__(self):
        return f"Move({self.target})"


class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(degrees)"
    description = "Rotate by specified degrees (0, 90, 180, 270)"
    example = "Rotate(90)"
    VALID_DEGREES = [0, 90, 180, 270]
    
    def __init__(self, degrees=None):
        super().__init__(degrees)
        self.degrees = int(degrees) if degrees else None

    def rotate_agent(self, room, degrees: int):
        """Rotate agent by specified degrees, shift coordinate system to keep agent at origin"""
        rotation_matrix = self._get_rotation_matrix(degrees)
        for obj in room.objects:
            if obj.name != room.agent.name:
                obj.pos = obj.pos @ rotation_matrix
                obj.ori = obj.ori @ rotation_matrix
    
    def format_pattern(self) -> str:
        return r"^Rotate\(([0-9-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Rotated by {self.degrees}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only {self.VALID_DEGREES} allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute rotate action on room state."""
        if self.degrees not in self.VALID_DEGREES:
            return ActionResult(False, self.get_feedback(False, "invalid_degree"))
        
        self.rotate_agent(room, self.degrees)
            
        return ActionResult(True, self.get_feedback(True), {'degrees': self.degrees})
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ReturnAction(BaseAction):
    """Return to anchor position"""
    
    format_desc = "Return()"
    description = "Return to the starting anchor position"
    example = "Return()"
    
    def format_pattern(self) -> str:
        return r"^Return\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Returned to anchor."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot return to anchor: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute return action on room state."""
        agent_anchor = kwargs.get('agent_anchor')
        if not agent_anchor:
            return ActionResult(False, self.error_message())
        
        ori_to_deg = {(0, 1): 0, (0, -1): 180, (1, 0): 90, (-1, 0): 270}
        target_deg = ori_to_deg[tuple(agent_anchor.ori)]
        
        move_action = MoveAction()
        move_action.move_agent_to_pos(room, agent_anchor.pos)
        
        rotate_action = RotateAction()
        rotate_action.rotate_agent(room, target_deg)
        
        return ActionResult(True, self.get_feedback(True), {'target_name': agent_anchor.name, 'degrees': target_deg})
    
    def __repr__(self):
        return "Return()"


class ObserveAction(BaseAction):
    """Observe spatial relationships of all objects in view"""
    
    format_desc = "Observe()"
    description = "Observe spatial relationships of all objects in the field of view relative to your current position"
    example = "Observe()"
    
    def __init__(self):
        super().__init__()
    
    def format_pattern(self) -> str:
        return r"^Observe\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Observed: {kwargs.get('answer', 'N/A')}"
    
    def error_message(self, error_type: str) -> str:
        return "Cannot observe: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute observe action on room state."""
        neglect_objects = kwargs.get('neglect_objects', [])
        visible_objects = [obj for obj in room.objects if self._is_visible(room.agent, obj) and obj.name not in neglect_objects]
        
        if not visible_objects:
            answer = "Nothing to observe in the current field of view."
            return ActionResult(True, self.get_feedback(True, answer=answer), {
                'answer': answer, 'visible_objects': [], 'relationships': []
            })

        relationships = []
        for obj in visible_objects:
            _, dir_str = room.get_direction(obj.name, room.agent.name, perspective='ego')
            relationships.append(f"{obj.name} is {dir_str}")

        answer = ", ".join(relationships) + "."
        
        return ActionResult(True, self.get_feedback(True, answer=answer), {
            'answer': answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships
        })
    
    def is_final(self) -> bool:
        return True
    
    def __repr__(self):
        return "Observe()"


class TermAction(BaseAction):
    """Terminate exploration"""
    
    format_desc = "Term()"
    description = "Terminate the exploration phase"
    example = "Term()"
    
    def format_pattern(self) -> str:
        return r"^Term\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute term action on room state."""
        return ActionResult(True, self.get_feedback(True), {'terminated': True})
    
    def is_final(self) -> bool:
        return True
    
    def is_term(self) -> bool:
        return True
    
    def __repr__(self):
        return "Term()"


class QueryAction(BaseAction):
    """Query spatial relationship of a specific object"""
    
    format_desc = "Query(object_name)"
    description = "Query spatial relationship of a specific object relative to your current position"
    example = "Query(table)"
    
    def __init__(self, target=None):
        super().__init__(target)
        self.target = target
    
    def format_pattern(self) -> str:
        return r"^Query\(([A-Za-z0-9_-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Queried: {kwargs.get('answer', 'N/A')}"
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "not visible"}
        return f"Cannot query '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute query action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"))
        
        target_obj = room.get_object_by_name(self.target)
        if not self._is_visible(room.agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"))
        
        dir_pair, dir_str = room.get_direction(self.target, room.agent.name, perspective='ego')
        answer = f"{self.target} is {dir_str}"
        
        return ActionResult(True, self.get_feedback(True, answer=answer), {
            'answer': answer, 'target_object': self.target, 
            'direction_pair': dir_pair, 'direction_string': dir_str
        })
    
    def is_final(self) -> bool:
        return True
    
    def __repr__(self):
        return f"Query({self.target})"


# Action registry for easy lookup
ACTION_CLASSES = [MoveAction, RotateAction, ReturnAction, ObserveAction, TermAction, QueryAction]


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        return f"ActionSequence(motions=[{motions}], final={self.final_action})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        """Parse action string into ActionSequence"""
        parts = action_str.split(';')
        if len(parts) > 2:
            return None
            
        motion_actions = []
        
        if len(parts) == 2:
            for item in [i.strip() for i in parts[0].split(',') if i.strip()]:
                action = cls._parse_single_action(item)
                if not action or action.is_final():
                    return None
                motion_actions.append(action)
        
        final_action = cls._parse_single_action(parts[-1].strip())
        if not final_action or not final_action.is_final():
            return None
            
        if isinstance(final_action, TermAction) and motion_actions:
            return None
            
        return cls(motion_actions, final_action)
    
    @staticmethod
    def _parse_single_action(action_str: str) -> Optional[BaseAction]:
        """Parse a single action string using registered action classes"""
        for action_class in ACTION_CLASSES:
            if action := action_class.parse(action_str):
                return action
        return None
    
    @staticmethod
    def get_usage_instructions() -> str:
        """Get usage instructions for action sequences"""
        motion_actions = [cls for cls in ACTION_CLASSES if not cls().is_final()]
        final_actions = [cls for cls in ACTION_CLASSES if cls().is_final()]
        
        instructions = (
            "## Action Format\n"
            "Use semicolon to separate movement actions from final query/term action.\n"
            "Multiple movements can be chained with commas.\n\n"
            "## Available Actions\n"
        )
        
        if motion_actions:
            instructions += "### Movement Actions\n"
            instructions += "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in motion_actions)
            instructions += "\n\n"
        
        if final_actions:
            instructions += "### Final Actions\n"
            instructions += "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in final_actions)
            instructions += "\n\n"
        
        instructions += (
            "## Examples\n"
            f"Simple observation: {ObserveAction.example}\n"
            f"Move then observe: {MoveAction.example}; {ObserveAction.example}\n"
            f"Multiple moves: {MoveAction.example}, {RotateAction.example}; {ObserveAction.example}\n"
            f"Return to start: {ReturnAction.example}; {ObserveAction.example}\n"
            f"Terminate: {TermAction.example}\n\n"
            "## Rules\n"
            "- Last action must be Observe() or Term()\n"
            "- Term() cannot have movement actions before it\n"
            "- You have a field of view of 90 degrees, 45 to the left and 45 to the right\n"
            "- You can only move to objects that are within your field of view\n"
        )
        
        return instructions 