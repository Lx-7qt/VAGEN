from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import re
import numpy as np

"""
Base action definitions and common functionality.
Contains the abstract base class and result types for all actions.
"""


@dataclass
class ActionResult:
    """Result of action execution"""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAction(ABC):
    """Base class for all actions"""
    
    # Class attributes to be overridden by subclasses
    format_desc = ""
    description = ""
    example = ""
    
    def __init__(self, parameters=None):
        self.parameters = parameters
    
    @abstractmethod
    def format_pattern(self) -> str:
        """Return regex pattern for parsing this action"""
        pass
    
    @abstractmethod
    def success_message(self, **kwargs) -> str:
        """Return success message for this action"""
        pass
    
    @abstractmethod
    def error_message(self, error_type: str) -> str:
        """Return error message for this action"""
        pass
    
    @abstractmethod
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute action on room state.
        
        Args:
            room: Room to execute action on
            **kwargs: Additional execution context (e.g., coordinate system info)
            
        Returns:
            ActionResult containing success status, message, and additional data
        """
        pass
    
    def _is_visible(self, from_obj, to_obj) -> bool:
        """Check if to_obj is visible from from_obj (180-degree field of view)."""
        direction_vec = to_obj.pos - from_obj.pos
        if np.allclose(direction_vec, 0):
            return True
        direction_norm = direction_vec / np.linalg.norm(direction_vec)
        ori_norm = from_obj.ori / np.linalg.norm(from_obj.ori)
        return np.dot(direction_norm, ori_norm) >= -1e-6
    
    def _get_rotation_matrix(self, degrees: int) -> np.ndarray:
        """Get rotation matrix for specified degrees.
        NOTE agent rotates clockwise <==> other object rotates counterclockwise
        """
        rotations = {0: [[1,0],[0,1]], 90: [[0,1],[-1,0]], 180: [[-1,0],[0,-1]], 270: [[0,-1],[1,0]]}
        return np.array(rotations[degrees])
    
    def is_final(self) -> bool:
        """Check if this is a final action (ends the sequence)"""
        return False
    
    def is_term(self) -> bool:
        """Check if this is a termination action"""
        return False
    
    @classmethod
    def parse(cls, action_str: str):
        """Parse action string and return instance if matches"""
        instance = cls()
        if match := re.match(instance.format_pattern(), action_str):
            return cls(*match.groups())
        return None

    def get_feedback(self, success: bool, error_type: str = None, **kwargs) -> str:
        """Generate feedback based on execution result"""
        return self.success_message(**kwargs) if success else self.error_message(error_type) 