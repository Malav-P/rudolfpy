"""Position measurement object"""

import numpy as np
from ._base_measurement import BaseMeasurement

class MeasurementPosition(BaseMeasurement):
    """Position vector measurement object"""
    def __init__(self):
        super().__init__()
        return
    
    def predict_measurement(self, t, x):
        return x[0:3]
    
    def measurement_partials(self, t, x):
        return np.concatenate((np.eye(3), np.zeros((3,3))), axis=1)