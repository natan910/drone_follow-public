"""Ready-made toy worlds for demos and tests."""

from typing import Optional, Tuple

from sim.virtual_world import Box, VirtualWorld, room


def demo_world(person: Optional[Tuple[float, float]] = (-7.0, 4.0), seed: int = 1,
               **kwargs) -> VirtualWorld:
    """A 20 x 14 m room with a pillar, a box, and a partition wall."""
    boxes = room(20, 14) + [Box(2, 0, 4, 2), Box(-5, -3, -4, -2), Box(-2, 3, 3, 3.4)]
    return VirtualWorld(boxes=boxes, person=person, start=(0.0, 0.0, 30.0), seed=seed, **kwargs)
