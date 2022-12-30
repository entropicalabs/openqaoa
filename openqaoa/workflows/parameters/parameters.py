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

from openqaoa.problems.helper_functions import convert2serialize


class Parameters:

    def clean_attributes(self):

        #delete all attributes that are == to None
        for key in list(self.__dict__.keys()):
            if self.__dict__[key] is None:
                del self.__dict__[key]

    def asdict(self):
        return convert2serialize(self)
