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

from openqaoa.utilities import convert2serialize


class Parameters:
    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            yield (key[1:] if key.startswith("_") else key, value)

    def asdict(self):
        return convert2serialize(dict(self))

    @classmethod
    def from_dict(cls, dictionary={}):

        obj = cls()
        for key, value in dictionary.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj
