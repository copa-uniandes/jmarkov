from typing import Dict, List, Callable

import numpy as np

from jmarkov.state import State, StatesSet
from jmarkov.event import Event, EventsSet
from jmarkov.transition import TransitionsSet


class LabeledMatrix:
    """
    Takes a numpy matrix with dimentions nxn with the labels of its states.
    Consults like Q[0][0] are allowed.
    Shows rows and columns labeled with the description of each state.
    """

    def __init__(self, matrix: np.ndarray, labels: list, name: str = "Q"):
        """
        matrix : np.ndarray
            A numpy matrix with dimentions nxn.
        lables : list of str
            One label per state in idex order.
        name : str
            The name of the matrix that will be showed in the printed header.
        """
        self._matrix = matrix
        self._labels = labels
        self._name   = name

    def __getitem__(self, index):
        """Allows the Q[0][0] consults"""
        return self._matrix[index]

    def __len__(self):
        return len(self._matrix)

    def __iter__(self):
        return iter(self._matrix)

    def __str__(self):
        n         = len(self._matrix)
        col_w     = max(10, max(len(lb) for lb in self._labels) + 2)
        row_lbl_w = max(len(lb) for lb in self._labels) + 2

        header = " " * row_lbl_w + "".join(lb.center(col_w) for lb in self._labels)
        sep    = " " * row_lbl_w + "-" * (col_w * n)

        lines = [f"\nMatriz {self._name} ({n}x{n}):", header, sep]

        for i, row in enumerate(self._matrix):
            row_str = self._labels[i].ljust(row_lbl_w)
            for v in row:
                row_str += f"{v:.2f}".rjust(col_w)
            lines.append(row_str)

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


class MarkovProcess:
    """
    Base class for building and analysing a finite continuous-time Markov chain
    (CTMC).
 
    Subclasses must override active_transitions() to define the specific model
    transition logic. Once generate() is called, the state space is explored
    automatically via the BuildRS algorithm, and the rate matrix R and the
    generator matrix Q become available.
    """

    def __init__(self, states_list: List[State], events_list: List[Event]):
        """
        states_list : List of State
            Initial state(s) for the BuildRS exploration.
        events_list : List of Event
            All the events that can trigger transitions in the model. 
        """
        self._events = EventsSet(events_list)
        self._states = StatesSet(states_list)

        self._unchecked: List[State] = []
        self._rates: Dict[State, TransitionsSet] = {}

        self._rates_matrix     = None
        self._generator_matrix = None

    # ------------------------------------------------------------------
    # Metodo abstracto
    # ------------------------------------------------------------------

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        """
        Given an State and a Event, returns a TransitionSet of the transitions that are active.
        If the Event does not apply to the State it returns an empty TransitionSet

        This method must be overriden by the user in every problem. 
        """
        raise NotImplementedError("You must implement active_transitions in your subclass.")

    # ------------------------------------------------------------------
    # Algoritmo BuildRS
    # ------------------------------------------------------------------

    def generate(self, key: Callable = None):
        """
        Builds the state space using the BuildRs algorithm

        All initial state(s) are enqueued first, which allows the algorithm to support the two cases, there is only one initial state or
        we already know hoy many states we have.

        After exploration, numerate_states() is called to assign the final index, and the user can define a custom order by using key. 
        """
        in_queue = set()
        for state in self._states.get_states():
            self._unchecked.append(state)
            in_queue.add(state)

        explored = set()

        while self._unchecked:

            current = self._unchecked.pop(0)
            in_queue.discard(current)

            if current.is_terminal():
                explored.add(current)
                continue

            total_transitions = TransitionsSet()

            for event in self._events:
                trans = self.active_transitions(current, event)

                if trans:
                    for t in trans:
                        destination = t.get_destination()
                        rate        = t.get_rate()

                        if self._states.contains(destination):
                            destination = self._states.get_canonical(destination)
                        else:
                            self._states.add(destination)
                            self._unchecked.append(destination)
                            in_queue.add(destination)

                        total_transitions.add_rate(destination, rate)

            self._rates[current] = total_transitions
            explored.add(current)

        self._states.numerate_states(key=key)

    # ------------------------------------------------------------------
    # Etiquetas internas (ordenadas por indice)
    # ------------------------------------------------------------------

    def _get_labels(self) -> list:
        """
        Returns string descriptions of the State objects sorted by index
        It is used in LabeledMatrix.
        """
        states_sorted = sorted(self._states.get_states(), key=lambda s: s.get_index())
        return [s.description() for s in states_sorted]

    # ------------------------------------------------------------------
    # Matriz R
    # ------------------------------------------------------------------

    def get_rates_matrix(self) -> LabeledMatrix:
        """
        Builds the rates matrix (R) and returns it as a LabeledMatrix
        """
        n = self._states.size()
        matrix = np.zeros((n, n))

        for state in self._states:
            i = state.get_index()
            transitions = self._rates.get(state)
            if transitions:
                for t in transitions:
                    j = t.get_destination().get_index()
                    matrix[i][j] += t.get_rate()

        self._rates_matrix = matrix
        return LabeledMatrix(matrix, self._get_labels(), name="R")

    # ------------------------------------------------------------------
    # Matriz Q
    # ------------------------------------------------------------------

    def get_generator_matrix(self) -> LabeledMatrix:
        """
        Builds the generator matrix (Q) and returns it as a LabeledMatrix
        """
        if self._rates_matrix is None:
            self.get_rates_matrix()

        n = self._states.size()
        matrix = np.zeros((n, n))

        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._rates_matrix[i][j]
                    row_sum += matrix[i][j]
            matrix[i][i] = -row_sum

        self._generator_matrix = matrix
        return LabeledMatrix(matrix, self._get_labels(), name="Q")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_states(self) -> StatesSet:
        """Returns the state space as a StatesSet"""
        return self._states

    def get_num_states(self) -> int:
        """Returns the number of states"""
        return self._states.size()

    def print_states(self):
        "Prints all the states sorted by index"
        print(f"\nEspacio de estados ({self._states.size()} estados):")
        for s in sorted(self._states.get_states(), key=lambda s: s.get_index()):
            term = " (terminal)" if s.is_terminal() else ""
            print(f"  [{s.get_index()}] {s.description()}{term}")


