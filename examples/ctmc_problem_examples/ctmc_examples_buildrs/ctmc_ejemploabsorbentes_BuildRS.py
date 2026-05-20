from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class absorbentes(MarkovProcess):
    """
    Example with 2 state variables where x(t) = Is the number of engines running on the left wing, 
    y(t) = Is the number of engines running on the right wing. Where Sx = {0, 1, 2},S(y) = {0, 1, 2}.
    Where any state 0 is an absorbent state.
    """

    lam:float

    def __init__(self,lam):
        self.lam=lam



        super().__init__(states_list=[State(2,2)],
                         events_list=[Event(description="FallaIzquierda"),
                                      Event(description="FallaDerecha")])
        
    def active_transitions(self, state, event)-> TransitionsSet:
        i=state.get_value(0)
        j=state.get_value(1)
        trans=TransitionsSet()

        if event.description()=="FallaIzquierda":
            if i>0:
                trans.add(State(i-1,j,terminal=(i-1==0)),i*self.lam)

        if event.description()=="FallaDerecha":
            if j>0:
                trans.add(State(i,j-1,terminal=(j-1==0)),j*self.lam) 

        return trans  
    

modelo=absorbentes(lam = 2)
modelo.generate(key=lambda s:(s.get_value(0),s.get_value(1)))
 
modelo.print_states()
Q=modelo.get_generator_matrix()
print(Q)
print(Q[3][0]*Q[4][1])
