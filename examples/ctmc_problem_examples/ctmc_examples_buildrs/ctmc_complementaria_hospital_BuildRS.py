import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))

from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class hospital(MarkovProcess):
    """
    Example with 2 state variables where x(t) = Number of patients of type 1, 
    y(t) = Number of patients of type 2. Where Sx = {0, 1, 2,..., B},
    S(y) = {0, 1, 2, ..., B-b}
    """

    lam1:float
    lam2:float
    mu1:float
    mu:float
    B:int
    b:int

    def __init__(self, lam1,lam2,mu1,mu2,B,b):
        self.lam1 = lam1
        self.lam2 = lam2
        self.mu1 = mu1
        self.mu2 = mu2
        self.B = B
        self.b = b
        

        super().__init__(
            states_list = [State(0,0)],
            events_list = [
                Event("LlegaTipo1"),
                Event("LlegaTipo2"),
                Event("SaleTipo1"),
                Event("SaleTipo2"),
            ]
        )

    def active_transitions(self, state, event):
        trans=TransitionsSet()
        i=state.get_value(0)
        j=state.get_value(1)

        if event.description()=="LlegaTipo1":
            if i+j<self.B:
                trans.add(State(i+1,j),self.lam1)

        if event.description()=="LlegaTipo2":
            if (i+j<self.B) and (j<self.B-self.b):
                trans.add(State(i,j+1),self.lam2)

        if event.description()=="SaleTipo1":
            if i>0:
                trans.add(State(i-1,j),self.mu1*i)
        if event.description()=="SaleTipo2":
            if j>0:
                trans.add(State(i,j-1),self.mu2*j)       
        return trans

    
modelo=hospital(lam1 = 6.5, lam2 = 3, mu1 = 2, mu2 = 1.5, B = 3, b = 1)
modelo.generate(key=lambda s:(s.get_value(0)))
Q=modelo.get_generator_matrix()
print(Q)
modelo.print_states()
print(modelo.get_num_states())
print(Q[2][5])
