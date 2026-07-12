from typing import Dict, Iterator

from jmarkov.state import State


class Transition:
    """
    Represents a single transition between two states with the corresponding rate.
    """
    def __init__(self, destination: State, rate: float):
        """
        destination : State
            The reached State after the transition is active.
        rate : float
            The rate at which the destination can be reached.
        """
        self._destination = destination
        self._rate = rate

    def get_destination(self) -> State:
        """Returns the destination state of the transition."""
        return self._destination

    def get_rate(self) -> float:
        """Returns the transition rate."""
        return self._rate

    def label(self) -> str:
        """Returns an string representation of the transition"""
        return f"{self._destination.label()} -> {self._rate}"

    def __str__(self):
        return self.label()

class TransitionsSet:
    """
    Set of the outgoing transitions from an single state.

    Uses dict where:
        key   -> State (destiny State)
        value -> rate  (accumulated rate to that destiny)
    """

    def __init__(self):
        self._transitions: Dict[State, float] = {}

    def add(self, state: State, rate: float) -> bool:
        """Adds using destination state and rate. Returns TRUE if it is new."""
        is_new = state not in self._transitions
        self._transitions[state] = rate
        return is_new

    def add_rate(self, state: State, rate: float) -> float:
        """
        Accumulates the rate to a destination State. It is used when there are multiple Events that lead to the destination from the current State.
        Returns the rate that was stored before; if it is new it will return 0.0.
        """
        old_rate = self._transitions.get(state, 0.0)
        self._transitions[state] = old_rate + rate
        return old_rate

    def get_rate(self, state: State) -> float:
        """Returns the accumulated rate to the destination State given."""
        return self._transitions.get(state, 0.0)

    def size(self) -> int:
        """Return the number destination states."""
        return len(self._transitions)

    # Boolean soport for 'if transitions_set:' in markovprocess.py
    def __bool__(self) -> bool:
        return len(self._transitions) > 0

    def __iter__(self) -> Iterator[Transition]:
        for state, rate in self._transitions.items():
            yield Transition(state, rate)

    def __str__(self):
        if self.size() > 15:
            return f"TransitionsSet with {self.size()} transitions"
        return str(
            [f"{state.label()} -> {rate}"
             for state, rate in self._transitions.items()]
        )