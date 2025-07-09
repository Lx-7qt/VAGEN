# Spatial Gym

A text-based spatial reasoning environment where agents explore rooms and answer spatial relationship questions.

## Overview

The Spatial Gym enables agents to:
- Explore spatial environments through movement and observation
- Learn spatial relationships between objects
- Answer spatial reasoning questions based on their exploration

## Quick Start

```python
from ragen.env.spatial import SpatialGym

# Create and use the environment
env = SpatialGym(config)
observation = env.reset()

# Agent explores
action = "Move(table)"
observation, reward, done, info = env.step(action)

# Agent observes
action = "Observe()"
observation, reward, done, info = env.step(action)

# End exploration
action = "Terminate()"
observation, reward, done, info = env.step(action)
```

## Exploration Modes

### Active Exploration
- Agent has 90° field of view
- Can only query visible objects
- Must move strategically to discover all relationships

### Semi-Active Exploration  
- Agent can ask about any two objects
- More flexible than active mode

### Passive Exploration
- System provides complete exploration history
- No agent movement required

## Available Actions

- `Move(object)` - Move to a specific object
- `Rotate(degrees)` - Rotate by specified degrees
- `Query(object)` - Ask about object's relationship to agent
- `Observe()` - Observe current surroundings
- `Return()` - Return to starting position
- `Terminate()` - End exploration phase

## Evaluation Tasks

The environment includes various spatial reasoning tasks:
- **Direction**: Understanding cardinal directions (N, S, E, W)
- **Rotation**: Understanding rotational relationships  
- **Point of View**: Spatial relationships from different perspectives
- **Object Relations**: Relative positions between objects

## Configuration

Environment behavior is controlled through `config.py`. See the Base module documentation for detailed technical information and extension guidelines.
