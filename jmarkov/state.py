from typing import Dict, Iterable, Any, Callable


class State:
    """
    Represents a single state in a continuous time markov chain.
    The state is defined by an arbitrary number of values that are stored as a tuple.
    """

    def __init__(self, *args, terminal: bool = False):
        """
        *args : any
            The variable(s) that represent/identify the created state.
        terminal : Bool
            Indicates if the state is an absorbing state (True) or not an absorbing state (False)    
        """
        self._index: int = -1  #StatesSet.numerate_states() changes this value
        self._terminal: bool = terminal
        self._values: tuple = args
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self._terminal, self._values) == (other._terminal, other._values)

    def __hash__(self):
        return hash((self._terminal, self._values))

    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (self._terminal, self._values) < (other._terminal, other._values)
    
    #State Properties

    def is_terminal(self) -> bool:
        """Indicates if the state is an absorbing state (True) or not an absorbing state (False)"""
        return self._terminal

    def get_index(self) -> int:
        """Returns the index of the state. If the state has not been numbered yet it returns -1."""
        return self._index

    def set_index(self, i: int):
        """Sets the index of the state.Called in StatesSet"""
        self._index = i

    def get_values(self) -> tuple:
        """Returns the tuple(values) of this state."""
        return self._values

    def get_value(self, i: int = 0):
        """Returns the value stored in the position i of the tuple.By default it takes the first one."""
        return self._values[i]

    def num_variables(self) -> int:
        """Returns the number of state variables."""
        return len(self._values)

    def description(self) -> str:
        """Returns the state variables"""
        if len(self._values) == 1:
            return str(self._values[0])
        return str(self._values)

    def label(self) -> str:
        """Label that including the index. Used in Transition"""
        return f"State {self._index}"

    def __str__(self):
        return self.description()


class StatesSet:
    """Collection of State objects. """
    def __init__(self, states: Iterable[Any] = None):
        """
        states:Initial state(s) to populate the set with.
        """
        self._states: Dict[Any, Any] = {}
        self._closed = False
        self._index = 0
        self._num_variables = None   

        if states:
            for s in states:
                self.add(s)

    def add(self, state: State):
        """
        Adds a state to the set. 
        """
        if self._closed:
            raise Exception("The state set is closed. No more states can be added.")

        #All the states must have the same number of variables. The first state sets the number of variables to compare.
        if self._num_variables is None:
            self._num_variables = state.num_variables()
        elif state.num_variables() != self._num_variables:
            raise ValueError(
                f"Cannot add state {state.get_values()}: "
                f"it has {state.num_variables()} variable(s), "
                f"but the model uses {self._num_variables}. "
                f"All states must have the same number of variables."
            )

        self._states[state] = state
        state.set_index(self._index)
        self._index += 1

    def contains(self, state: State) -> bool:
        """
        Returns True if an equal state already exists in the set.
        """
        return state in self._states

    def get_canonical(self, state: State) -> State:
        """
        Returns the state object stored in the set that is equal to the given state.This avoids duplicates.
        """
        return self._states[state]

    def numerate_states(self, key: Callable = None):
        """
        Assigns the index to all states and close the set.

        key: Sorting function applied to each State.

             Examples:
               key=None -> discovery order
               key=lambda s: (s.get_value(1), s.get_value(0)) -> first y, then x

        Returns the total number of states.
        """
        if key is not None:
            ordered = sorted(self._states.values(), key=key)
        else:
            ordered = list(self._states.values())

        for i, state in enumerate(ordered):
            state.set_index(i)

        self._closed = True
        return len(self._states)

    def size(self) -> int:
        """Returns the number of states."""
        return len(self._states)

    def get_states(self) -> list:
        """Returns a list of the all State objects in the order of insertion."""
        return list(self._states.values())

    def __iter__(self):
        return iter(self._states.values())

    def __str__(self):
        return str([s.description() for s in self._states.values()])