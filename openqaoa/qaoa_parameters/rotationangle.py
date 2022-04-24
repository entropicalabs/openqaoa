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
