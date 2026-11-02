"""Shared tools for task-related operations in agents."""

from agents import FunctionTool, function_tool

from scenesmith.agent_utils.room import RoomScene


def get_task_tagger_tools(scene: RoomScene) -> list[FunctionTool]:
    """Create tools for tagging task-relevant objects.

    Args:
        scene: RoomScene to operate on.

    Returns:
        List containing the tag_as_task_object tool.
    """

    @function_tool
    def tag_as_task_object(object_id: str) -> str:
        """Mark a placed object as a task-relevant object for RL training.

        Call this immediately after placing any object marked [TASK OBJECT]
        in the prompt.

        Args:
            object_id: The ID returned by the placement tool.

        Returns:
            Confirmation message.
        """
        if object_id not in scene.objects:
            return f"Error: object '{object_id}' not found in scene."
        scene.objects[object_id].metadata["is_task_object"] = True
        return f"Tagged '{object_id}' as task object."

    return [tag_as_task_object]
