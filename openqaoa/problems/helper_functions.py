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

def convert2serialize(obj):
    if isinstance(obj, dict):
        return {k: convert2serialize(v) for k, v in obj.items() if v is not None}
    elif hasattr(obj, "_ast"):
        return convert2serialize(obj._ast())
    elif not isinstance(obj, str) and hasattr(obj, "__iter__"):
        return [convert2serialize(v) for v in obj if v is not None]
    elif hasattr(obj, "__dict__"):
        return {
            k: convert2serialize(v)
            for k, v in obj.__dict__.items()
            if not callable(v) and v is not None
        }
    else:
        return obj


