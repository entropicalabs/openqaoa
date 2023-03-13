from importlib.metadata import entry_points

def plugin_finder_dict() -> list:
    
    available_plugins = entry_points()['openqaoa.plugins']
    
    output_dict = dict()
    for each_plugin_entry_point in available_plugins:
        try:
            output_dict[each_plugin_entry_point.name] = each_plugin_entry_point.load()
        except ModuleNotFoundError:
            print("The {} module has not been installed.".format(each_plugin_entry_point.name))
        except AttributeError:
            print("An error has occured when trying to attach the {} plugin.".format(each_plugin_entry_point.name))
    
    return output_dict