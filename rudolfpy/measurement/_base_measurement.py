"""Base measurement object"""

class BaseMeasurement:
    def __init__(self):
        return
    
    def predict_measurement(self, t, x):
        raise NotImplementedError("predict_measurement method must be implemented in subclass")
    
    def measurement_partials(self, t, x):
        raise NotImplementedError("measurement_partials method must be implemented in subclass")