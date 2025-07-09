import gymnasium as gym
import numpy as np
import re
import os
import json
from typing import Optional
from PIL import Image

from vagen.env.spatial.config import SpatialGymConfig
from vagen.env.spatial.Base import (
    EvaluationManager,
    Room,
    ActionSequence,
    ExplorationManager,
    generate_room,
    MoveAction,
    RotateAction,
    ReturnAction,
    ObserveAction,
    TermAction
)
from vagen.env.spatial.Base.utils.room_utils import initialize_room_from_json
from vagen.env.spatial.utils.generate_history import AutoExplore


instruction = (
    "# Spatial Mapping Task\n"
    "\n"
    "You are exploring a room to discover spatial relationships between objects.\n"
    "Build a complete mental map by finding where each object is relative to others.\n"
    "\n"
    "## Spatial Relationships\n"
    "When you query an object, you get its position relative to you: (horizontal, vertical)\n"
    "\n"
    "- Horizontal: left, right, same\n"
    "- Vertical: front, back, same\n"
    "- Example: (left, front) means object is to your left and in front of you\n"
    "\n"
    "## Key Points\n"
    "- Relationships are relative: if A is left of B, then B is right of A\n"
    "- Terminate when you have enough information to map all object pairs\n"
    "\n"
    "## Room Layout\n"
    "{room_info}\n"
    "\n"
    "{exp_history}\n"
    "\n"
    "{exp_answer_format}\n"
)


def _vector_to_dir(ori: np.ndarray) -> str:
    """Convert orientation vector to cardinal direction string."""
    vec = tuple(np.round(ori).astype(int))
    mapping = {(0, 1): 'north', (1, 0): 'east', (0, -1): 'south', (-1, 0): 'west'}
    return mapping.get(vec, 'north')


class SpatialGym(gym.Env):
    metadata = {'render.modes': ['multi_modal']}

    def __init__(self, data_dir, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.is_exp_stage = None
        self.max_exp_steps = None


        # Load metadata JSON
        json_dir = os.path.join(data_dir, "meta_data.json")
        self.current_data = json.load(open(json_dir, 'r'))
        self.image_dir = data_dir

        # Preload images into lookup map
        self.image_map = {}
        for entry in self.current_data.get('images', []):
            key = (entry['position'], entry['direction'])
            filename = entry['filename']
            # Ensure .png extension
            if not filename.lower().endswith('.png'):
                filename = f"{filename}.png"
            path = os.path.join(self.image_dir, filename)
            img = Image.open(path)
            self.image_map[key] = img

        # Spaces (handled via multi_modal_data externally)
        self.observation_space = None
        self.action_space = gym.spaces.Discrete(1)

        # State trackers
        self.room_s_0 = None
        self.room_s_t = None
        self.room_s_end = None
        self.exploration_manager = None
        self.evaluation_manager = None
        self.n_novel_queries = 0
        self.n_valid_queries = 0
        self.current_position = 'original'
        self.current_direction = 'north'

    def _create_observation(self) -> dict:
        """Return dict with text and the single correct image."""
        key = (self.current_position, self.current_direction)
        img = self.image_map[key]  # must exist

        return {
            'filename': key,
            'multi_modal_data': img
        }

    def reset(self, seed: int = None):
        super().reset(seed=seed)
        self.max_exp_steps = self.config.max_exp_steps
        self.n_novel_queries = 0
        self.n_valid_queries = 0

        # Initialize state
        self.room_s_0 = initialize_room_from_json(self.current_data)
        self.room_s_t = self.room_s_0.copy()
        self.is_exp_stage = (self.config.exp_type != 'passive')
        if self.is_exp_stage:
            self.exploration_manager = ExplorationManager(self.room_s_0)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random)

        # Initial viewer state
        self.current_position = 'original'
        self.current_direction = _vector_to_dir(self.room_s_0.agent.ori)

        return self._create_observation(), {}

    def step(self, action: str):
        reward = 0
        done = False
        info = {}

        if self.is_exp_stage:
            self.max_exp_steps -= 1
            seq = ActionSequence.parse(action)
            if not seq:
                return self._create_observation(), -0.1, False, {}

            # enforce final action type
            final = seq.final_action
            if not isinstance(final, (ObserveAction, TermAction)):
                return self._create_observation(), -0.1, False, {}

            msg, exp_info = self.exploration_manager.execute_action_sequence(seq)

            # Update viewer state by replaying all motion actions
            for act in seq.motion_actions:
                if isinstance(act, MoveAction):
                    self.current_position = act.target
                elif isinstance(act, RotateAction):
                    self.current_direction = {0:'north', 90:'east', 180:'south', 270:'west'}[act.degrees]
            # Handle return
            if isinstance(final, ReturnAction):
                self.current_position = 'original'
                self.current_direction = _vector_to_dir(self.room_s_0.agent.ori)

            if isinstance(final, TermAction) or self.max_exp_steps < 0:
                self.is_exp_stage = False
            return self._create_observation(), reward, done, info
        else:
            correct, reward, eval_info = self.evaluation_manager.evaluate_answer(action)
            done = not self.evaluation_manager.next_task()
            return self._create_observation(), reward, done, eval_info

    def render(self, mode=None):
        return self._create_observation()

    def get_env_info(self) -> dict:
        return {
            'config': self.config.to_dict(),
            'room_s_0': self.room_s_0.to_dict(),
            'room_s_t': self.room_s_t.to_dict(),
            'room_s_end': self.room_s_end.to_dict() if self.room_s_end else None,
        }

    def get_exp_efficiency(self) -> dict:
        if self.exploration_manager:
            return self.exploration_manager.get_exploration_efficiency()
        return {}

    def get_eval_performance(self) -> dict:
        if self.evaluation_manager:
            return self.evaluation_manager.get_evaluation_summary()
        return {}

if __name__ == "__main__":

    # TODO
    def test_passive_exploration():
        """Test passive exploration mode."""
        print("Testing Passive Exploration...")
        
        config = SpatialGymConfig(
            exp_type='passive',
            n_objects=3,
            room_range=[-5, 5],
            eval_tasks=[
                {"task_type": "dir", "task_kwargs": {}},
                {"task_type": "all_pairs", "task_kwargs": {}}
            ],
            max_exp_steps=50
        )
        path = os.path.join(
        os.path.dirname(__file__),
            "my_output/"
        )
        env = SpatialGym(path, config)
        obs, info = env.reset(seed=42)
        print(f"room: {env.room_s_0}")
        print(f"Initial observation <<{obs}>>")
        print(f"Contains exploration history: {'Exploration History' in obs}")
        
        # Simulate evaluation answers
        done = False
        step_count = 0
        while not done and step_count < 10:
            # Simple answer format for testing
            answer = "(unknown, unknown)"
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step(answer)
            step_count += 1
            print(f"observation <<{obs}>>, Step {step_count}: Reward={reward}, Done={done}")
        
        # Check evaluation performance
        eval_perf = env.get_eval_performance()
        print(f"Evaluation accuracy: {eval_perf['accuracy']:.2f}")
        print("Passive exploration test completed.\n")

    def test_active_exploration():
        """Test active exploration mode."""
        print("Testing Active Exploration...")
        
        config = SpatialGymConfig(
            exp_type='active',
            n_objects=4,
            room_range=[-8, 8],
            eval_tasks=[{"task_type": "dir", "task_kwargs": {}}],
            max_exp_steps=20
        )
        path = os.path.join(
        os.path.dirname(__file__),
            "my_output/"
        )

        env = SpatialGym(path, config)
        obs, info = env.reset(seed=123)
        print(f"room: {env.room_s_0}")
        print(f"Initial observation contains action format: {'Available Actions' in obs}")
        
        # Test exploration phase
        exploration_actions = [

            "Rotate(90); Observe()",
            "Rotate(180); Observe()",
            "Rotate(90);Move(blue_side_chair_3386958);Observe()",
            "Observe()"
        ]
        
        step_count = 0
        for action in exploration_actions:
            if env.is_exp_stage:
                obs, reward, done, info = env.step(action)
                step_count += 1
                print(f"Observation <<{obs}>>, Exploration step {step_count}: Action='{action}', Valid response received")
                if not env.is_exp_stage:
                    print("Transitioned to evaluation phase")
                    break
            else:
                break

        print(f"all objects in exploration manager: {env.exploration_manager.exploration_room.all_objects}")
        print(f"Exploration graph: {env.exploration_manager.exp_graph.to_dict()}")
        
        # Test evaluation phase
        if not env.is_exp_stage:
            answer = "right"
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step(answer)
            print(f"Evaluation answer: Reward={reward}, Done={done}")
        
        # Check exploration efficiency
        exp_eff = env.get_exp_efficiency()
        print(f"Exploration coverage: {exp_eff['coverage']:.2f}")
        print(f"Novel queries: {exp_eff['n_novel_queries']}/{exp_eff['n_valid_queries']}")
        print("Active exploration test completed.\n")

    def test_different_generation_types():
        """Test different room generation types."""
        print("Testing Different Generation Types...")
        
        generation_types = ["rand", "rot", "a2e", "pov"]
        
        for gen_type in generation_types:
            try:
                print(f"Testing generation type: {gen_type}")
                
                # Adjust perspective based on generation type
                perspective = "ego" if gen_type in ["rot", "pov"] else "ego"
                
                config = SpatialGymConfig(
                    generation_type=gen_type,
                    perspective=perspective,
                    exp_type='passive',
                    n_objects=3,
                    eval_tasks=[{"task_type": "dir", "task_kwargs": {}}]
                )
                
                env = SpatialGym(config)
                obs, info = env.reset(seed=42)
                print(f"room: {env.room_s_0}")
                
                # Get environment info
                env_info = env.get_env_info()
                print(f"  Room generated with {len(env_info['room_s_0']['all_objects'])} objects")
                print(f"  Generation type: {env_info['config']['generation_type']}")
                
            except Exception as e:
                print(f"  Error with {gen_type}: {e}")

        print("Generation types test completed.\n")

    def test_evaluation_tasks():
        """Test different evaluation task types."""
        print("Testing Different Evaluation Tasks...")
        
        task_configs = [
            {"task_type": "dir", "task_kwargs": {}},
            {"task_type": "rot", "task_kwargs": {"turn_direction": "clockwise"}},
            {"task_type": "pov", "task_kwargs": {}},
            {"task_type": "all_pairs", "task_kwargs": {}}
        ]
        
        for task_config in task_configs:
            try:
                print(f"Testing task: {task_config['task_type']}")
                
                config = SpatialGymConfig(
                    exp_type='passive',
                    n_objects=3,
                    eval_tasks=[task_config],
                    perspective='ego',
                    generation_type='pov'
                )
                
                env = SpatialGym(config)
                obs, info = env.reset(seed=42)
                print(f"room: {env.room_s_0}")
                print(f"observation: {obs}")
                
                # Try one evaluation step
                answer = env.evaluation_manager._get_current_eval_task().answer
                print(f"ground truth answer: {answer}")
                obs, reward, done, info = env.step(answer)
                print(f"  Task executed successfully, reward: {reward}")
                
            except Exception as e:
                print(f"  Error with {task_config['task_type']}: {e}")
        
        print("Evaluation tasks test completed.\n")

    def test_action_parsing():
        """Test action sequence parsing."""
        print("Testing Action Parsing...")
        
        from vagen.env.spatial.Base import ActionSequence
        
        test_actions = [
            "Query(table)",
            "Move(chair), Rotate(90); Query(table)",
            "Rotate(90); Query(table)",
            "Return(); Query(table)",
            "Term()",
            "Invalid action",
            "Query(table) Move(chair)",  # Multiple actions
            ""
        ]
        
        for action_str in test_actions:
            action_seq = ActionSequence.parse(action_str)
            if action_seq:
                print(f"  '{action_str}' -> Valid: {action_seq}")
            else:
                print(f"  '{action_str}' -> Invalid")
        
        print("Action parsing test completed.\n")

    def test_environment_states():
        """Test environment state transitions."""
        print("Testing Environment States...")
        
        config = SpatialGymConfig(
            exp_type='active',
            n_objects=3,
            max_exp_steps=5
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        
        print(f"Initial state - Is exploration: {env.is_exp_stage}")
        
        # Force transition to evaluation by terminating
        obs, reward, done, info = env.step("Term()")
        print(f"After termination - Is exploration: {env.is_exp_stage}")
        
        # Test evaluation phase
        if not env.is_exp_stage:
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step("left")
            print(f"Evaluation step completed - Done: {done}")
        
        # Check final states
        env_info = env.get_env_info()
        print(f"Room states available - s_0: {bool(env_info['room_s_0'])}, "
              f"s_t: {bool(env_info['room_s_t'])}, s_end: {bool(env_info['room_s_end'])}")
        
        print("Environment states test completed.\n")

    def test_configuration_validation():
        """Test configuration validation."""
        print("Testing Configuration Validation...")
        
        # Test valid configurations
        valid_configs = [
            {"exp_type": "passive", "perspective": "ego"},
            {"exp_type": "active", "perspective": "ego"},
            {"generation_type": "rand", "perspective": "ego"},
            {"generation_type": "rot", "perspective": "ego"}
        ]
        
        for config_dict in valid_configs:
            try:
                config = SpatialGymConfig(**config_dict)
                print(f"  Valid config: {config_dict}")
            except Exception as e:
                print(f"  Unexpected error with {config_dict}: {e}")
        
        # Test invalid configurations
        invalid_configs = [
            {"generation_type": "invalid_type"},
            {"exp_type": "invalid_exp"},
            {"perspective": "invalid_perspective"},
            {"generation_type": "rot", "perspective": "allo"}  # Incompatible combination
        ]
        
        for config_dict in invalid_configs:
            try:
                config = SpatialGymConfig(**config_dict)
                print(f"  Unexpected success with invalid config: {config_dict}")
            except Exception as e:
                print(f"  Expected error with {config_dict}: {type(e).__name__}")
        
        print("Configuration validation test completed.\n")

    # Run all tests
    print("="*50)
    print("SPATIAL GYM ENVIRONMENT TESTS")
    print("="*50)
    
    try:
        # test_passive_exploration()
        test_active_exploration()
        # test_different_generation_types()
        #test_evaluation_tasks()
        # test_action_parsing()
        # test_environment_states()
        # test_configuration_validation()
        
        print("="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()