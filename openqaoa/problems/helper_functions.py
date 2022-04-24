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


def check_kwargs(list_expected_params, list_default_values, **kwargs):
    """
    Checks that the given list of expected parameters can be found in the
    kwargs given as input. If so, it returns the parameters from kwargs, else
    it raises an exception.

    Args:
        list_expected_params: List[str]
            List of string containing the name of the expected parameters in 
            kwargs
        list_default_values: List
            List containing the deafult values of the expected parameters in 
            kwargs
        **kwargs:
            Keyword arguments where keys are supposed to be the expected params

    Returns:
        A tuple with the actual expected parameters if they are found in kwargs.

    Raises:
        ValueError: 
            If one of the expected arguments is not found in kwargs and its 
            default value is not specified.
    """
    def check_kwarg(expected_param, default_value, **kwargs):
        param = kwargs.pop(expected_param, default_value)

        if param is None:
            raise ValueError(
                f"Parameter '{expected_param}' should be specified")

        return param

    params = []
    for expected_param, default_value in zip(list_expected_params, list_default_values):
        params.append(check_kwarg(expected_param, default_value, **kwargs))

    return tuple(params)
