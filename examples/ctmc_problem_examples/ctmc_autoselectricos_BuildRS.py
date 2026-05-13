
from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class autos(MarkovProcess):

        
    def __init__(self,lam, mu, alpha):
        self.lam   = lam
        self.mu    = mu
        self.alpha = alpha


        super().__init__(
            states_list=[State(0, 0)],
            events_list=[Event(description="llegada"),
                         Event(description="servicio_recarga"),
                         Event(description="servicio_cambio")
                         ])

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        trans = TransitionsSet()
        i = state.get_value(0)
        j = state.get_value(1)

        #Llegadas
        if event.description() == "llegada":
            #Llegada a estacion de recarga
            if i == 0:
                trans.add(State(1, j), 0.7 * self.lam)
            #Llegada a estacion de cambio
            if j == 0:
                trans.add(State(i, 1), 0.3 * self.lam)

        #Batería recargada
        elif event.description() == "servicio_recarga":
            if i == 1:
                trans.add(State(0, j), self.mu)
                
        #Cambio de Batería       
        elif event.description() == "servicio_cambio":
            if j == 1:
                trans.add(State(i, 2), self.alpha)
            elif j == 2:
                trans.add(State(i, 0), self.alpha)

        return trans



modelo = autos(lam=3, mu=2, alpha=5)
modelo.generate(key=lambda s: (s.get_value(0)))

modelo.print_states()


Q = modelo.get_generator_matrix()
print(Q)
