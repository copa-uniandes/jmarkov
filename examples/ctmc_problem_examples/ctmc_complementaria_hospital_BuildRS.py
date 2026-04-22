from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class hospital(MarkovProcess):

    def __init__(self, lam1,lam2,mu1,mu2,B,b,):
        self.lam1 = lam1
        self.lam2 = lam2
        self.mu1 = mu1
        self.mu2 = mu2
        self.B = B
        self.b = b
        

        super().__init__(
            states_list = [State(0,0)],
            events_list = [
                Event("llegatipo1"),
                Event("llegatipo2"),
                Event("saletipo1"),
                Event("saletipo2"),
            ]
        )

    def active_transitions(self, state, event):
        trans=TransitionsSet()
        i=state.get_value(0)
        j=state.get_value(1)

        if event.description()=="llegatipo1":
            if i+j<self.B:
                trans.add(State(i+1,j),self.lam1)

        if event.description()=="llegatipo2":
            if (i+j<self.B) and (j<self.B-self.b):
                trans.add(State(i,j+1),self.lam2)

        if event.description()=="saletipo1":
            if i>0:
                trans.add(State(i-1,j),self.mu1*i)
        if event.description()=="saletipo2":
            if j>0:
                trans.add(State(i,j-1),self.mu2*j)       
        return trans
    
modelo=hospital(6.5,3,2,1.5,3,1)
modelo.generate(key=lambda s:(s.get_value(0)))
Q=modelo.get_generator_matrix()
print(Q)
modelo.print_states()
print(modelo.get_num_states())
print(Q[2][5])
