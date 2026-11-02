"""Abstract base class for programmatic randomization operations."""

from abc import ABC, abstractmethod

from scenesmith.agent_utils.room import RoomScene


class BaseRandomizationOp(ABC):
    """Base class for all programmatic scene randomization operations."""

    @abstractmethod
    def apply(
        self,
        scene: RoomScene,
        params: dict,
        locked_ids: set[str],
    ) -> RoomScene:
        """Apply the randomization operation to the scene.

        Args:
            scene: RoomScene to modify in place.
            params: Op-specific parameters.
            locked_ids: Object IDs that must not be modified (task-critical).

        Returns:
            The modified scene (same object, mutated in place).
        """

    @abstractmethod
    def validate(self, scene: RoomScene) -> list[str]:
        """Validate the scene after applying the operation.

        Args:
            scene: Scene to validate.

        Returns:
            List of violation messages. Empty list means the scene is valid.
        """
