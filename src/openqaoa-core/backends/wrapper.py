class TwirlingWrapper:
    
    def __init__(self, backend: Backend, n_batches):
        self.backend = backend
        self.n_batches = n_batches
        
    def __getattr__(self, name):
        return getattr(self.backend, name)
    
    def expectation(self, ):
        return 2