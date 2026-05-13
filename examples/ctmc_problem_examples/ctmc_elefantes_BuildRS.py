from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet

class elefantes(MarkovProcess):

    def __init__(self,lam,mu,theta,omega):
        self.lam=lam
        self.mu=mu
        self.theta=theta
        self.omega=omega

        super().__init__(states_list=[State(0,"W")],
                         events_list=[Event(description="Llegada"),
                                      Event(description="Salida"),
                                      Event(description="Agotada"),
                                      Event(description="Lista")])

    def active_transitions(self, state, event)-> TransitionsSet:
        trans=TransitionsSet()
        i = state.get_value(0)
        j = state.get_value(1)

        #Llegada de clientes
        if event.description()=="Llegada":
            if i<6:
                trans.add(State(i+1,j),self.lam)

        #Salida de clientes
        elif event.description()=="Salida":
            if i>0 and j=="W":
                trans.add(State(i-1,j),self.mu)

        #Agotamiento del elefante
        elif event.description()=="Agotada":
            if j=="W":
                trans.add(State(i,"R"),self.theta)
            
        #Recuperacion del elefante
        elif event.description()=="Lista":
            if j=="R":
                trans.add(State(i,"W"),self.omega)
        return trans
    
modelo = elefantes(lam=0.5,mu=1/4,theta=1/45,omega=1/10)
modelo.generate(key=lambda s:(s.get_value(0)))
Q = modelo.get_generator_matrix()
print(Q)
modelo.print_states()
