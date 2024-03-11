"""
Data transformations to be used
"""
class DataTransform(ABC):
    def __init__(self):
        self.pointwise = False
    @abstractmethod
    def fit(self, array:np.array, split:str) -> None:
        pass
    @abstractmethod
    def transform(self, array:np.array, split:str) -> np.array:
        pass
    @abstractmethod
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        pass
    def fit_transform(self, array:np.array, split:str) -> np.array:
        self.fit(array, split)
        return self.transform(array, split)

class GausNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.var = None
    def fit(self, array:np.array, split:str) -> None:
        self.mean = np.mean(array)
        self.var = np.std(array)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.mean) / self.var
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * self.var + self.mean

class RangeNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.lower = None
        self.upper = None
    def fit(self, array:np.array, split:str) -> None:
        self.lower = np.min(array)
        self.upper = np.max(array)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.lower) / (self.upper - self.lower)
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * (self.upper - self.lower) + self.lower

class PassNorm(DataTransform):
    def __init__(self):
        super().__init__()
    def fit(self, array:np.array, split:str) -> None:
        pass
    def transform(self, array:np.array, split:str) -> np.array:
        return array
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array