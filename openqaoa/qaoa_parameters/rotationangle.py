#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import annotations
from typing import List, Union, Callable

class RotationAngle(object):

    def __init__(self, angle_relationship: Callable, gate_label: GateMapLabel,
                 value: Union[int, float] = None):
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
