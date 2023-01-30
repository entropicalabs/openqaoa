from typing import List, Union, Callable


class RotationAngle(object):

    def __init__(self, angle_relationship: Callable, pauli_label: List,
                 pauli_angle: Union[int, float] = None):

        self._angle = angle_relationship
        self.pauli_label = pauli_label
        self.pauli_angle = pauli_angle

    @property
    def rotation_angle(self) -> Union[int, float]:

        return self._angle(self.pauli_angle)
