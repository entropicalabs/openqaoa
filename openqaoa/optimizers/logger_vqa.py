from typing import Union, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from copy import deepcopy
    
class UpdateMethod(ABC):
    
    @classmethod
    @abstractmethod
    def update(self):
        
        pass
    
    
class LoggerVariable(object):
    
    def __init__(self, attribute_name: str, history_update_method: UpdateMethod, best_update_method: UpdateMethod):
        
        self.name = attribute_name
        self.best = []
        self.history = []
        
        self.history_update_method = history_update_method
        self.best_update_method = best_update_method
        
    def update(self, new_value):
        
        self.update_history(new_value)
        self.update_best(new_value)
    
    def update_history(self, new_value):
        
        return self.history_update_method.update(self, 'history', new_value)
    
    def update_best(self, new_value):
        
        return self.best_update_method.update(self, 'best', new_value)


class AppendValue(UpdateMethod):
    
    def update(self, attribute_name: str, new_value: Union[list, int, float, str]) -> None:
        
        get_var = getattr(self, attribute_name)
        get_var.append(new_value)
        setattr(self, attribute_name, get_var)
    

class ReplaceValue(UpdateMethod):
    
    def update(self, attribute_name: str, new_value: Union[list, int, float, str]) -> None:
        
        setattr(self, attribute_name, [new_value])

        
class EmptyValue(UpdateMethod):
    
    def update(self, attribute_name: str, new_value: Union[list, int, float, str]) -> None:
        
        setattr(self, attribute_name, []) 


class IfLowerDo(UpdateMethod):
    
    def __init__(self, update_method: UpdateMethod):
        
        self._update_method = update_method
    
    def update(self, logger_variable: LoggerVariable, attribute_name: str, new_value: Union[int, float]) -> None:
        
        try:
            old_value = getattr(logger_variable, attribute_name)[-1]
            if new_value < old_value:
                self._update_method.update(logger_variable, attribute_name, new_value)
        except IndexError:
            AppendValue.update(logger_variable, attribute_name, new_value)
            
            
class IfHigherDo(UpdateMethod):
    
    def __init__(self, update_method: UpdateMethod):
        
        self._update_method = update_method
    
    def update(self, logger_variable: LoggerVariable, attribute_name: str, new_value: Union[int, float]) -> None:
        
        try:
            old_value = getattr(logger_variable, attribute_name)[-1]
            if new_value > old_value:
                self._update_method.update(logger_variable, attribute_name, new_value)
        except IndexError:
            AppendValue.update(logger_variable, attribute_name, new_value)
        

class LoggerVariableFactory(object):
    
    """
    Creates Logger Variable Objects
    """
    
    history_bool_mapping = {'True': AppendValue, 'False': EmptyValue}
    best_string_mapping = {'HighestOnly': IfHigherDo(ReplaceValue), 
                           'HighestSoFar': IfHigherDo(AppendValue), 
                           'LowestOnly': IfLowerDo(ReplaceValue), 
                           'LowestSoFar': IfLowerDo(AppendValue), 
                           'Replace': ReplaceValue, 
                           'Append': AppendValue}
    
    @classmethod
    def create_logger_variable(self, attribute_name: str, history_update_bool: bool, best_update_string: str) -> LoggerVariable:
        
        history_update_method = self.history_bool_mapping[str(history_update_bool)]
        best_update_method = self.best_string_mapping[best_update_string]
        
        return LoggerVariable(attribute_name, history_update_method, best_update_method)
    
class Logger(object):
    
    def __init__(self, initialisation_variables: dict, logger_update_structure: dict):

        for attribute_name, attribute_properties in initialisation_variables.items():
            self._create_variable(attribute_name = attribute_name, 
                                  history_update_bool = attribute_properties['history_update_bool'], 
                                  best_update_string = attribute_properties['best_update_string'])
        self._create_logger_update_structure(logger_update_structure['root_nodes'], 
                                             logger_update_structure['best_update_structure'])
    
    def _create_variable(self, attribute_name: str, history_update_bool: bool, 
                         best_update_string: str):
        
        """
        If a variable was created was this method, ensure that the logger update
        strucuture is updated to include the new variable. Else the best value
        of the new variable will not be updated.
        """
        
        setattr(self, attribute_name, 
                LoggerVariableFactory.create_logger_variable(
                    attribute_name, history_update_bool, best_update_string))
    
    def _create_logger_update_structure(self, 
                                        root_nodes: List[str],
                                        best_update_relations: Tuple[List[str]]):
        
        """
        This method creates an internal representation based on the way in which 
        the Logger Variables should be updated using a NetworkX Directed Graph.
        
        Parameter
        ---------
        root_nodes: `list`
            Starting points of the update graphs. The name of the Logger Variable 
            that needs to be updated first should be named in this list.
        best_update_relations: `list`
            If the updating of B depends on A, (A changes, Update B). This 
            relationship should be encapsuled/represented with the following 
            list: ['A', 'B'].
        """
        
        self.best_update_structure = nx.DiGraph()
        
        self.root_nodes = root_nodes
        self.update_edges = best_update_relations
        
    @property
    def update_edges(self):
        
        return self._update_edges
    
    @update_edges.setter
    def update_edges(self, update_edges: Tuple[List[str]]):
        
        self.best_update_structure.add_edges_from(update_edges)
        self._update_edges = update_edges
        
        return self._update_edges
    
    @property
    def root_nodes(self):
        
        return self._root_nodes
    
    @root_nodes.setter
    def root_nodes(self, root_nodes: List[str]):
        
        self.best_update_structure.add_nodes_from(root_nodes)
        self._root_nodes = root_nodes
        
        return self._root_nodes
    
    def log_variables(self, input_dict: dict):
        
        """
        Updates all the histories of the logger variables.
        Best values of the logger variables are only updated according to the rules 
        specified by its update structure. If the attribute is not in the update
        structure it isnt updated.
        
        input_dict:
            Should contain a dictionary with the key as the name of the 
            attribute to be updated and its respective value. Note that the
            attribute should have been created beforehand.
        """
        
        for each_key in input_dict.keys():
            self._log_history(each_key, input_dict[each_key])
        
        # Updates based on best_update_structure, the networkx graph.
        change_dict = dict()
        
        node_list = deepcopy(self.root_nodes)
        
        while len(node_list) != 0:

            mid_list = []
            for each_node in node_list:
                
                change_bool = False
                
                # Checks if a nodes predecessor has been updated. If not all
                # of them were updated, the node is skipped.
                node_predecessor = self.best_update_structure.pred[each_node]
                updated_pred_count = np.sum([change_dict[each_precessor_node] for each_precessor_node in node_predecessor])
                if len(node_predecessor) != updated_pred_count:
                    continue
                        
                # Update node if attribute name is in input dictionary and 
                # its predecessors hsa been updated.
                if each_node in input_dict.keys():
                    logged_var = getattr(self, each_node)
                    old_best = logged_var.best.copy()
                    self._log_best(each_node, input_dict[each_node])
                    new_best = logged_var.best
                    if type(new_best[0]) == np.ndarray:
                        if not np.array_equal(new_best, old_best):
                            change_bool = True
                    elif new_best != old_best:
                        change_bool = True
                
                # Keeps track of the nodes whose best values have changed
                change_dict[each_node] = change_bool
                
                # Retrieve the next set of nodes to update if the current node
                # was changed.
                if change_bool == True:
                    mid_list.extend(self.best_update_structure.adj[each_node].keys())
                    
            node_list = list(set(mid_list))
    
    
    def _log_history(self, attribute_name: str, attribute_value):
        
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_history(attribute_value)
        
    def _log_best(self, attribute_name: str, attribute_value):
        
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_best(attribute_value)