from abc import ABCMeta

class Env_interface(metaclass=ABCMeta):
    def render():
        pass
    def reset():
        pass
    def step():
        pass