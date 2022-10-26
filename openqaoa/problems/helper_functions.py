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


def convert_binary_to_ising(terms, weights):
    """
    Converts the weights from a [0, 1] encoding to an Ising problem [-1, 1] 0 is mapped to +1 and 1 to -1 respectively

    Parameters
    ----------
    terms: list[list]
        terms of the hamiltonian
    weights: list[float]

    Returns
    -------
    terms_weights: tuple(list[list],list[float])
        Tuple containing the converted list of terms and list of weights
    """
    new_terms_weights = []
    constant = 0

    for i, term in enumerate(terms):
        if len(term) == 1:
            new_terms_weights.append((term, -0.5 * weights[i]))
            constant += 0.5 * weights[i]
        elif len(term) == 2:
            for t in term:
                new_terms_weights.append(([t], -0.25 * weights[i]))
            new_terms_weights.append((term, 0.25 * weights[i]))
            constant += 0.25 * weights[i]

    new_terms_weights.append(([], constant))

    return tuple(zip(*new_terms_weights))
