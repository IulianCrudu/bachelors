from typing import List
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from decimal import Decimal

import torch


IMG_SIZE = (1280, 720)
PX_ERROR = Decimal(20)


@dataclass
class Coordinate:
    x: float
    y: float

    def is_equal_with_error(self, coordinate: "Coordinate") -> bool:
        if not(Decimal(coordinate.x) - PX_ERROR <= Decimal(self.x) <= Decimal(coordinate.x) + PX_ERROR):
            return False

        if not(Decimal(coordinate.y) - PX_ERROR <= Decimal(self.y) <= Decimal(coordinate.y) + PX_ERROR):
            return False

        return True


@dataclass
class ImageInfo:
    path: str
    coordinates: List[torch.Tensor]


@dataclass
class Car:
    top_left_corner: Coordinate
    bottom_right_corner: Coordinate
    id: UUID = field(default_factory=uuid4)
