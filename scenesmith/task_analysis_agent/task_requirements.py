"""RL task requirements extracted by the TaskAnalysisAgent.

Holds the structured output of the task analysis step and provides helpers
for building the augmented scene prompt and RL task context string used
throughout the pipeline.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass, field
from pathlib import Path

console_logger = logging.getLogger(__name__)


@dataclass
class RequiredObject:
    """A single object that must be present in the generated scene."""

    name: str
    """Human-readable name, e.g. 'ceramic coffee mug'."""

    is_task_object: bool
    """True if the agent directly interacts with this object during the task."""

    properties: dict
    """Task-relevant properties, e.g. {'approx_size_cm': [10, 8, 12], 'must_be_upright': True}."""

    placement_hint: str
    """Where to place this object, e.g. 'on dining table surface'."""

    placement_stage: str = "furniture"
    """Which pipeline stage places this object.

    "furniture"  — placed by the furniture agent (tables, countertops, chairs, shelves).
    "manipuland" — placed by the manipuland agent (mugs, plates, tools, small graspables).
    """

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_task_object": self.is_task_object,
            "placement_stage": self.placement_stage,
            "properties": self.properties,
            "placement_hint": self.placement_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RequiredObject":
        is_task_object = data.get("is_task_object", False)
        default_stage = "manipuland" if is_task_object else "furniture"
        return cls(
            name=data["name"],
            is_task_object=is_task_object,
            placement_stage=data.get("placement_stage", default_stage),
            properties=data.get("properties", {}),
            placement_hint=data.get("placement_hint", ""),
        )


@dataclass
class DomainRandomizationSpec:
    """Describes the domain randomization space for this task.

    Inferred from task semantics by the TaskAnalysisAgent. Defines what can
    vary in the scene without breaking the RL task. At artifact export time,
    names are replaced with actual scene object IDs by the RLArtifactAgent.
    """

    task_object_variants: list[dict] = field(default_factory=list)
    """Per-task-object DR: color/material/shape substitutes, position range on surface.

    Example entry:
        {"name": "mug", "can_swap_with": ["ceramic cup", "glass"],
         "randomizable_properties": ["color", "material"],
         "position_range": "anywhere on parent surface"}
    """

    furniture_movable: list[dict] = field(default_factory=list)
    """Furniture pieces that can be repositioned/swapped as a whole unit.

    Example entry:
        {"name": "dining table", "can_reposition": True,
         "can_swap_with": ["kitchen table"],
         "constraint": "must remain in same room, task objects still accessible"}
    """

    clutter_objects: list[dict] = field(default_factory=list)
    """Non-task objects on the same surfaces that can be added/removed.

    Example entry:
        {"surface": "dining table", "can_add": True, "can_remove": True,
         "density_range": ["sparse", "moderate"]}
    """

    scene_level: dict = field(default_factory=dict)
    """Scene-level elements that can vary without breaking the task.

    Example:
        {"lighting_intensity": [0.5, 1.5], "floor_texture": True, "wall_color": True}
    """

    notes: str = ""
    """LLM reasoning about what cannot vary (e.g. 'goal location must remain fixed')."""

    def to_dict(self) -> dict:
        return {
            "task_object_variants": self.task_object_variants,
            "furniture_movable": self.furniture_movable,
            "clutter_objects": self.clutter_objects,
            "scene_level": self.scene_level,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainRandomizationSpec":
        return cls(
            task_object_variants=data.get("task_object_variants", []),
            furniture_movable=data.get("furniture_movable", []),
            clutter_objects=data.get("clutter_objects", []),
            scene_level=data.get("scene_level", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class TaskRequirements:
    """Structured requirements extracted from an RL task description.

    Produced by TaskAnalysisAgent at the start of the pipeline (before floor plan).
    Saved to task_requirements.json in the scene directory and reloaded on resume.

    The augmented prompt is injected into the house_prompt template variable seen
    by all designer and planner agents. The rl_task_context string is injected as
    a separate template variable into critic agents for RL Task Suitability scoring.
    """

    raw_prompt: str
    """The original RL task prompt as written by the user."""

    task_summary: str
    """One-line summary, e.g. 'Pick and place: mug from table to sink'."""

    task_type: str
    """Task category: 'pick_and_place' | 'navigation' | 'open_close' | 'sorting' | 'other'."""

    scene_description: str
    """Traditional scene description for the floor plan agent (no task-specific language)."""

    required_objects: list[RequiredObject]
    """Objects that MUST appear in the generated scene."""

    task_objects: list[str]
    """Names of objects the agent directly interacts with during the task."""

    spatial_constraints: list[str]
    """Spatial requirements critical for the task, e.g. 'sink must be accessible'."""

    scene_requirements: dict
    """High-level scene needs, e.g. {'room_types': ['kitchen'], 'needs_open_floor': True}."""

    rl_metadata: dict
    """RL training metadata, e.g. {'initial_state': '...', 'goal_state': '...'}."""

    domain_randomization: DomainRandomizationSpec
    """Full DR spec inferred from task semantics."""

    def build_augmented_prompt(self) -> str:
        """Build the full augmented prompt for designer/planner agents.

        Combines the clean scene description with RL task context so all
        agents naturally see what's required without structural YAML changes.
        Also includes an explicit instruction for the floor plan agent to
        propagate RL context into per-room prompts.
        """
        lines = [self.scene_description, "", "[RL Task Context - Read Carefully]"]
        lines.append(f"Task: {self.task_summary}")
        lines.append(f"Task Type: {self.task_type}")
        lines.append("")

        # Furniture objects: placed by the furniture agent.
        furniture_objs = [
            o for o in self.required_objects if o.placement_stage == "furniture"
        ]
        if furniture_objs:
            lines.append(
                "Furniture to Place (MUST be placed as scene furniture — these are NOT"
                " pre-installed structural or architectural elements, regardless of descriptive"
                ' language like "fixed" or "built-in"):'
            )
            for obj in furniture_objs:
                props_str = ", ".join(f"{k}: {v}" for k, v in obj.properties.items())
                lines.append(f"- {obj.name}: {props_str}, place {obj.placement_hint}")
            lines.append("")

        # Manipuland objects: placed by the manipuland agent (do not add as furniture).
        manipuland_objs = [
            o for o in self.required_objects if o.placement_stage == "manipuland"
        ]
        if manipuland_objs:
            lines.append(
                "Manipuland Objects (placed by the manipuland agent — do NOT place these as"
                " furniture):"
            )
            for obj in manipuland_objs:
                task_marker = "  [TASK OBJECT]" if obj.is_task_object else ""
                lines.append(f"- {obj.name}{task_marker}")
            lines.append("")

        # Spatial constraints.
        if self.spatial_constraints:
            lines.append("Spatial Constraints:")
            for c in self.spatial_constraints:
                lines.append(f"- {c}")
            lines.append("")

        # RL metadata.
        if self.rl_metadata.get("initial_state"):
            lines.append(f"Initial State: {self.rl_metadata['initial_state']}")
        if self.rl_metadata.get("goal_state"):
            lines.append(f"Success Criteria: {self.rl_metadata['goal_state']}")
        lines.append("")

        # Tagging instruction for furniture/manipuland designers.
        lines.append(
            "Agent Tagging: After placing any object marked [TASK OBJECT], "
            "immediately call tag_as_task_object(object_id) with the returned ID."
        )
        lines.append("")

        # Explicit room-prompt propagation instruction for the floor plan agent.
        lines.append(
            "Floor Plan Note: When generating per-room furniture prompts, "
            "include the full [RL Task Context] section verbatim in each room's prompt."
        )

        return "\n".join(lines)

    def build_rl_task_context_string(self) -> str:
        """Build a compact RL task context string for critic template variable.

        This is a shorter version of the full augmented prompt, containing only
        the information critics need to evaluate RL task suitability.
        """
        lines = [
            f"Task: {self.task_summary}",
            f"Task Type: {self.task_type}",
            "",
        ]
        # Furniture objects: critic MUST verify these are present in the scene.
        furniture_objs = [
            o for o in self.required_objects if o.placement_stage == "furniture"
        ]
        if furniture_objs:
            lines.append(
                "Required Furniture (MUST verify each is present in scene via"
                " get_current_scene_state):"
            )
            for obj in furniture_objs:
                lines.append(f"  - {obj.name} (place: {obj.placement_hint})")
            lines.append("")
        # Task objects placed by the manipuland agent.
        task_objs = [
            o for o in self.required_objects if o.placement_stage == "manipuland"
        ]
        if task_objs:
            lines.append("Required Task Objects (verified by manipuland agent):")
            for obj in task_objs:
                lines.append(f"  - {obj.name} (place: {obj.placement_hint})")
        lines.append("")
        if self.spatial_constraints:
            lines.append("Spatial Constraints:")
            for c in self.spatial_constraints:
                lines.append(f"  - {c}")
            lines.append("")
        if self.rl_metadata.get("goal_state"):
            lines.append(f"Success Criteria: {self.rl_metadata['goal_state']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "raw_prompt": self.raw_prompt,
            "task_summary": self.task_summary,
            "task_type": self.task_type,
            "scene_description": self.scene_description,
            "required_objects": [o.to_dict() for o in self.required_objects],
            "task_objects": self.task_objects,
            "spatial_constraints": self.spatial_constraints,
            "scene_requirements": self.scene_requirements,
            "rl_metadata": self.rl_metadata,
            "domain_randomization": self.domain_randomization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRequirements":
        """Deserialize from dict, with robust defaults for missing/malformed fields."""
        required_objects = [
            RequiredObject.from_dict(o) for o in data.get("required_objects", [])
        ]
        dr_data = data.get("domain_randomization", {})
        domain_randomization = DomainRandomizationSpec.from_dict(
            dr_data if isinstance(dr_data, dict) else {}
        )
        return cls(
            raw_prompt=data.get("raw_prompt", ""),
            task_summary=data.get("task_summary", ""),
            task_type=data.get("task_type", "other"),
            scene_description=data.get("scene_description", data.get("raw_prompt", "")),
            required_objects=required_objects,
            task_objects=data.get("task_objects", []),
            spatial_constraints=data.get("spatial_constraints", []),
            scene_requirements=data.get("scene_requirements", {}),
            rl_metadata=data.get("rl_metadata", {}),
            domain_randomization=domain_randomization,
        )

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console_logger.info(f"Task requirements saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "TaskRequirements":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
