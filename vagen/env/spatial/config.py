from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from omegaconf import ListConfig, OmegaConf
import os
from vagen.env.spatial.ToS_Base.tos_base import CANDIDATE_OBJECTS

@dataclass
class SpatialGymConfig:
    """
    Configuration for the SpatialGym environment.
    
    Parameters:
        room_range: Range for room dimensions
        n_objects: Number of objects in the room
        candidate_objects: List of objects that can be placed in the room
        generation_type: Type of room generation ('rand', 'rot', 'a2e', 'pov')
        exp_type: Exploration type ('passive', 'active')
        perspective: Perspective of exploration ('ego' or 'allo')
        eval_tasks: List of evaluation tasks with their configurations
        max_exp_steps: Maximum exploration steps for active exploration
        render_mode: Rendering mode (currently only 'text' supported)
    """
    # Room configuration
    room_range: List[int] = field(default_factory=lambda: [-10, 10])
    n_objects: int = 3
    candidate_objects: List[str] = field(default_factory=lambda: CANDIDATE_OBJECTS)
    generation_type: str = "rand"
    run = 0
    data_dir = os.path.join(
        os.path.dirname(__file__),
            f"output/run_0{run}/"
        )
    # Exploration configuration
    exp_type: str = 'passive'
    perspective: str = 'ego'
    max_exp_steps: int = 100
    
    # Evaluation configuration
    eval_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{"task_type": "dir", "task_kwargs": {}}])
    
    # Rendering configuration
    render_mode: str = "text"

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_generation_type()
        self._validate_perspective()
        self._validate_exp_type()
        self._validate_eval_tasks()
        self._validate_render_mode()
        self._validate_compatibility()

    def _validate_generation_type(self):
        """Validate generation_type parameter."""
        valid_types = ["rand", "rot", "a2e", "pov"]
        if self.generation_type not in valid_types:
            raise ValueError(f"generation_type must be one of {valid_types}")

    def _validate_perspective(self):
        """Validate perspective parameter."""
        valid_perspectives = ["ego", "allo"]
        if self.perspective not in valid_perspectives:
            raise ValueError(f"perspective must be one of {valid_perspectives}")

    def _validate_exp_type(self):
        """Validate exp_type parameter."""
        valid_exp_types = ["passive", "active"]
        if self.exp_type not in valid_exp_types:
            raise ValueError(f"exp_type must be one of {valid_exp_types}")

    def _validate_eval_tasks(self):
        """Validate eval_tasks parameter."""
        valid_eval_tasks = ["dir", "rot", "pov", "a2e", "e2a", "rev", "all_pairs"]

        if isinstance(self.eval_tasks, ListConfig):
            self.eval_tasks = OmegaConf.to_container(self.eval_tasks, resolve=True)
        
        if not self.eval_tasks:
            raise ValueError("eval_tasks must be non-empty")
        
        for i, task in enumerate(self.eval_tasks):
            if not isinstance(task, dict) or 'task_type' not in task:
                raise ValueError("Each eval_task must be a dict with 'task_type' key")
            
            task_type = task['task_type']
            if task_type not in valid_eval_tasks:
                raise ValueError(f"task_type '{task_type}' must be one of {valid_eval_tasks}")
            
            # Validate task-specific parameters
            task_kwargs = task.get('task_kwargs', {})
            self._validate_task_kwargs(task_type, task_kwargs)

    def _validate_task_kwargs(self, task_type: str, kwargs: Dict[str, Any]):
        """Validate task-specific parameters."""
        if task_type == 'dir':
            movement = kwargs.get('movement', 'static')
            valid_movements = ['static', 'object_move', 'agent_move', 'agent_turn']
            if movement not in valid_movements:
                raise ValueError(f"dir task movement must be one of {valid_movements}")
        
        elif task_type == 'rot':
            turn_direction = kwargs.get('turn_direction', 'clockwise')
            valid_directions = ['clockwise', 'counterclockwise']
            if turn_direction not in valid_directions:
                raise ValueError(f"rot task turn_direction must be one of {valid_directions}")

    def _validate_render_mode(self):
        """Validate render_mode parameter."""
        if self.render_mode != 'text':
            raise ValueError("Only 'text' rendering mode is currently supported")

    def _validate_compatibility(self):
        """Validate compatibility between different parameters."""
        # Check generation_type and perspective compatibility
        if self.generation_type in ['rot', 'pov'] and self.perspective != 'ego':
            raise ValueError("'rot' and 'pov' generation types only support 'ego' perspective")
        
        # Check eval_tasks and perspective compatibility
        task_types = [task['task_type'] for task in self.eval_tasks]
        
        if self.perspective == 'ego':
            if 'a2e' in task_types:
                raise ValueError("'a2e' task is only supported for allocentric exploration")
        else:  # perspective == 'allo'
            incompatible_tasks = [task for task in task_types if task in ['rot', 'pov', 'e2a']]
            if incompatible_tasks:
                raise ValueError(f"Tasks {incompatible_tasks} are not supported for allocentric perspective")

    def get_room_config(self) -> Dict[str, Any]:
        """Get configuration for room generation."""
        return {
            'room_range': self.room_range,
            'candidate_objects': self.candidate_objects,
            'generation_type': self.generation_type,
            'n_objects': self.n_objects,
            'perspective': self.perspective,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'room_range': self.room_range,
            'candidate_objects': self.candidate_objects,
            'generation_type': self.generation_type,
            'n_objects': self.n_objects,    
            'exp_type': self.exp_type,
            'perspective': self.perspective,
            'eval_tasks': self.eval_tasks,
            'max_exp_steps': self.max_exp_steps,
            'render_mode': self.render_mode,
        }
