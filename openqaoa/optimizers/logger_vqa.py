from typing import Union, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
    
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
    
    def update(self, logger_variable: LoggerVariable, attribute_name: str, new_value: Union[list, int, float, str]) -> None:
        
        try:
            old_value = getattr(logger_variable, attribute_name)[-1]
            if new_value < old_value:
                self._update_method.update(logger_variable, attribute_name, new_value)
        except IndexError:
            AppendValue.update(logger_variable, attribute_name, new_value)
            
            
class IfHigherDo(UpdateMethod):
    
    def __init__(self, update_method: UpdateMethod):
        
        self._update_method = update_method
    
    def update(self, logger_variable: LoggerVariable, attribute_name: str, new_value: Union[list, int, float, str]) -> None:
        
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
        self._create_logger_update_structure(logger_update_structure['best_update_structure'])
    
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
                                       best_update_structure: Tuple[List[str]]):
        
        """
        If the members in the tuple are in the same list, it means that their
        updates are independent of one another.
        If the members in the tuple are not in the same list, then the updates
        for the members in that list is dependent on whether any change was
        performed in all of members of the list before it.
        
        ([A], [B, C]) - If A is changed, update B and C
        ([A, B, C]) - Update A, B and C indepdently of each other.
        """
        
        self.best_update_structure = best_update_structure
    
    def log_variables(self, input_dict: dict):
        
        """
        Updates all the histories of the logger variables.
        Best values of the logger variables are only updated according to the rules 
        specified by its update structure.
        
        input_dict:
            Should contain a dictionary with the key as the name of the 
            attribute to be updated and its respective value. Note that the
            attribute should have been created beforehand.
        """
        
        for each_key in input_dict.keys():
            self._log_history(each_key, input_dict[each_key])
        
        # Updates based on the logging structure. Only moves to update the
        # new layer in the structure if the previous layer was updated and all
        # members of the layers were changed from the update.
        for each_index in range(len(self.best_update_structure)):
            change_count = 0
            for each_member in self.best_update_structure[each_index]:
                if each_member in input_dict.keys():
                    logged_var = getattr(self, each_member)
                    old_best = logged_var.best.copy()
                    self._log_best(each_member, input_dict[each_member])
                    new_best = logged_var.best
                    if type(new_best[0]) == np.ndarray:
                        if not np.array_equal(new_best, old_best):
                            change_count += 1
                    elif new_best != old_best:
                        change_count += 1
            if change_count != len(self.best_update_structure[each_index]):
                break
    
    def _log_history(self, attribute_name: str, attribute_value):
        
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_history(attribute_value)
        
    def _log_best(self, attribute_name: str, attribute_value):
        
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_best(attribute_value)