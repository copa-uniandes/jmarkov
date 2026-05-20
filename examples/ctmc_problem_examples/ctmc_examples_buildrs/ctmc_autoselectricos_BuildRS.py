
from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class autos(MarkovProcess):
    """
    Example with 2 state variables where x(t) = The state of the charging station, 
    y(t) = The state of the changing station. Where Sx = {Empty(0), Busy(1)},
    S(y) = {Empty(0), Busy at the first stage of the service(1), Busy at the second stage of the service(2)}
    """

    lam:float
    mu:float
    alpha:float
        
    def __init__(self,lam, mu, alpha):
        self.lam   = lam
        self.mu    = mu
        self.alpha = alpha


        super().__init__(
            states_list=[State(0, 0)],
            events_list=[Event(description="Llegada"),
                         Event(description="Servicio_Recarga"),
                         Event(description="Servicio_Cambio")
                         ])

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        trans = TransitionsSet()
        i = state.get_value(0)
        j = state.get_value(1)

        if event.description() == "Llegada":
            #Llegada a estacion de recarga
            if i == 0:
                trans.add(State(1, j), 0.7 * self.lam)
            #Llegada a estacion de cambio
            if j == 0:
                trans.add(State(i, 1), 0.3 * self.lam)

        #Batería recargada
        elif event.description() == "Servicio_Recarga":
            if i == 1:
                trans.add(State(0, j), self.mu)
                
        #Cambio de Batería       
        elif event.description() == "Servicio_Cambio":
            if j == 1:
                trans.add(State(i, 2), self.alpha)
            elif j == 2:
                trans.add(State(i, 0), self.alpha)

        return trans



modelo = autos(lam = 3, mu = 2, alpha = 5)
modelo.generate(key=lambda s: (s.get_value(0)))

modelo.print_states()


Q = modelo.get_generator_matrix()
print(Q)
