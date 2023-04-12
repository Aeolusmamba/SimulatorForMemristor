import abc


class layer(metaclass=abc.ABCMeta):  # abstract class

# property
    @property
    @abc.abstractmethod
    def out_dim(self):
        pass

    @out_dim.setter
    @abc.abstractmethod
    def out_dim(self, value):
        pass

    @property
    @abc.abstractmethod
    def in_dim(self):
        pass

    @in_dim.setter
    @abc.abstractmethod
    def in_dim(self, value):
        pass

    @property
    @abc.abstractmethod
    def weight(self):
        pass

    @weight.setter
    @abc.abstractmethod
    def weight(self, value):
        pass

    @property
    @abc.abstractmethod
    def in_history(self):
        pass

    @property
    @abc.abstractmethod
    def out_history(self):
        pass


# method
    @abc.abstractmethod
    def forward(self):  # the forward pass of the layer
        pass

    @abc.abstractmethod
    def backward(self, dy):  # the backward pass of the layer
        pass
