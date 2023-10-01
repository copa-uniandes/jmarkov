from abc import ABC, abstractmethod

class markov_chain(ABC):

    @property
    @abstractmethod
    def n_states(self):
        # number of states
        pass

    @property
    @abstractmethod
    def states(self):
        # markov chain states
        pass

    @abstractmethod
    def steady_state(self):
        # computes the steady state distribution
        pass

    @abstractmethod
    def first_passage_time(self, target:str):
        # computes the expected first passage time to target state
        pass

    @abstractmethod
    def occupation_time(self, nsteps:int):
        # computes the expected occupation time matrix in nsteps steps
        pass

    @abstractmethod
    def is_ergodic(self):
        # determines id the chain is ergodic or not
        pass

