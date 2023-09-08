from typing import List
import networkx as nx

from .basedevice import DeviceBase
from .plugin_finder import PLUGIN_DICT


def obtain_support_simulators() -> list:
    """This method returns a list containing the names of simulators supported
    by OQ. This list varies depending on the plugins available.
    """

    final_simulator_list = []
    for each_plugin_name, each_plugin_value in PLUGIN_DICT.items():
        if hasattr(each_plugin_value, "device_name_to_obj"):
            simulator_names = [
                each_key for each_key, _ in each_plugin_value.device_name_to_obj.items()
            ]
            final_simulator_list.extend(simulator_names)

    return final_simulator_list


SUPPORTED_LOCAL_SIMULATORS = [
    "vectorized",
    "analytical_simulator",
] + obtain_support_simulators()


class DeviceLocal(DeviceBase):
    """
    This class is a placeholder for all locally accessible devices.
    """

    def __init__(self, device_name: str):
        self.device_name = device_name
        self.device_location = "local"

    def check_connection(self) -> bool:
        if self.device_name in SUPPORTED_LOCAL_SIMULATORS:
            return True
        else:
            return False

    def connectivity(self, n_qubits: int) -> List[List[int]]:
        """
        The number of qubits for simulators depend on the problem
        """
        G = nx.complete_graph(n_qubits)
        return list(G.edges())
