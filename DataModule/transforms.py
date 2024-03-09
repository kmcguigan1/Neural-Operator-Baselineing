import numpy as np
from abc import ABC, abstractmethod

EPSILON = 1e-6

### These are the easier transforms that just return the 
# normalized data itself. These are applied over the whole dataset 
# meaning that we don't have to pass information ahead of time
class DataTransform(ABC):
    @abstractmethod
    def fit(self, array:np.array) -> None:
        pass
    @abstractmethod
    def transform(self, array:np.array) -> np.array:
        pass
    @abstractmethod
    def inverse_transform(self, array:np.array) -> np.array:
        pass
    def fit_transform(self, array:np.array) -> np.array:
        self.fit(array)
        return self.transform(array)

class GausNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.var = None
    def fit(self, array:np.array) -> None:
        self.mean = np.mean(array)
        self.var = np.std(array)
    def transform(self, array:np.array) -> np.array:
        return (array - self.mean) / self.var
    def inverse_transform(self, array:np.array) -> np.array:
        return array * self.var + self.mean

class RangeNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.lower = None
        self.upper = None
    def fit(self, array:np.array) -> None:
        self.lower = np.min(array)
        self.upper = np.max(array)
    def transform(self, array:np.array) -> np.array:
        return (array - self.lower) / (self.upper - self.lower)
    def inverse_transform(self, array:np.array) -> np.array:
        return array * (self.upper - self.lower) + self.lower 
        
def test():
    array = np.random.normal(loc=5.0, scale=3.0, size=(5,4,2,2))
    transform = InstanceGausNorm()
    transform.fit(array)
    transformed_array = transform.transform(array)
    print(np.mean(transformed_array, axis=(1,2,3)))
    print(np.std(transformed_array, axis=(1,2,3)))
    inversed_array = transform.inverse_transform(transformed_array)
    np.testing.assert_almost_equal(array, inversed_array)

if __name__ == '__main__':
    test()

