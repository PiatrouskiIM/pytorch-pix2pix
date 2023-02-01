from abc import abstractmethod
from abc import ABC


class GANObjective(ABC):
    @abstractmethod
    def practice_discrimination(self, generator, discriminator, data, device):
        raise NotImplementedError

    @abstractmethod
    def practice_generation(self, generator, discriminator, data, device):
        raise NotImplementedError
