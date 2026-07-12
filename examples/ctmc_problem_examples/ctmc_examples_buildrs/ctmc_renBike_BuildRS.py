import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))

from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class rentBike(MarkovProcess):
    """
    Example with 1 state variable where x(t) = Is the number of bicycles rented.
    Where Sx = {0, 1, 2,..., 6}.
    """

    tasallegada:float
    tasasalida:float


    def __init__(self, tasallegada,tasasalida):
        self.tasallegada=tasallegada
        self.tasasalida=tasasalida


        super().__init__(
            states_list=[State(0)],
            events_list=[Event(description="Llegada"),
                         Event(description="Salida")]
        )
    
    def active_transitions(self, state, event):
        trans=TransitionsSet()
        i=state.get_value(0)
        
        if event.description()=="Llegada":
            if i<6:
                trans.add(State(i+1),self.tasallegada)
        elif event.description()=="Salida":
            if i>0:
                trans.add(State(i-1),self.tasasalida*i)
        
        return trans
    
modelo=rentBike(tasallegada = 5, tasasalida = 9)
modelo.generate()
print(modelo.get_generator_matrix())
modelo.print_states()
  