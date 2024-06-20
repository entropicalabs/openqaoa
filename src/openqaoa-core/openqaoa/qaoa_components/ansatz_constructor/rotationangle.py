from __future__ import annotations
from typing import List, Union, Callable


class RotationAngle(object):
    def __init__(
        self,
        angle_relationship: Callable,
        gate_label: GateMapLabel,
        value: Union[int, float] = None,
    ):
        """
        Angle object as placeholder for assigning angles to the parameterized
        gates in the circuit

        Parameters
        ----------
        angle_relationship: `Callable`
            A function that takes input a parameter and assigns
            the angle to the gate depending on the relationship
            between the parameter and the angle
        gate_label: `GateMapLabel`
            The label for the gatemap object to which the rotationangle
            is assigned
        value: `int` or `float`
            Value of the parameter
        """

        self._angle = angle_relationship
        self.gate_label = gate_label
        self.value = value

    @property
    def rotation_angle(self) -> Union[int, float]:
        return self._angle(self.value)
