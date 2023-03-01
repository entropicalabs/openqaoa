def backend_boolean_list() -> list:
    
    try:
        import openqaoa_braket
    except ImportError:
        print("The braket module is not installed.")
        braket_bool = False
    else:
        braket_bool = True
        
    try:
        import openqaoa_azure
    except ImportError:
        print("The azure module is not installed.")
        azure_bool = False
    else:
        azure_bool = True
        
    try:
        import openqaoa_qiskit
    except ImportError:
        print("The qiskit module is not installed.")
        qiskit_bool = False
    else:
        qiskit_bool = True
        
    try:
        import openqaoa_pyquil
    except ImportError:
        print("The pyquil module is not installed.")
        pyquil_bool = False
    else:
        pyquil_bool = True
        
    
    return [braket_bool, azure_bool, qiskit_bool, pyquil_bool]