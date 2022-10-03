from abc import ABC, abstractmethod


class Attack(ABC):
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @abstractmethod
    def attack(self):
        pass
