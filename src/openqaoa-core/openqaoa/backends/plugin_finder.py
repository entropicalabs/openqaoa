from importlib.metadata import entry_points

def plugin_finder_dict() -> list:
    
    """
    This function searches the openqaoa.plugins entry point group to see if
    any of the openqaoa plugins have been installed. If they are, this function
    returns a dictionary whose keys are the names of the plugin libraries and
    whose values are the plugin's utilities module. 
    """
    
    output_dict = dict()
    
    try:
        available_plugins = entry_points()['openqaoa.plugins']
    
        for each_plugin_entry_point in available_plugins:
            try:
                output_dict[each_plugin_entry_point.name] = each_plugin_entry_point.load()
            except ModuleNotFoundError:
                print("The {} module has not been installed.".format(each_plugin_entry_point.name))
            except AttributeError:
                print("An error has occured when trying to attach the {} plugin.".format(each_plugin_entry_point.name))
            
    except KeyError:
        print("No plugins were found.")
    
    return output_dict

PLUGIN_DICT = plugin_finder_dict()