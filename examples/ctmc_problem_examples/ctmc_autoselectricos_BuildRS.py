
from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class autos(MarkovProcess):

    def __init__(self, lam, mu, alpha):
        self.lam   = lam
        self.mu    = mu
        self.alpha = alpha

        super().__init__(
            states_list=[State(0, 0)],
            events_list=[
                Event(description="llegada"),
                Event(description="servicio"),
                Event(description="transicion_j"),
            ]
        )

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        trans = TransitionsSet()
        i = state.get_value(0)
        j = state.get_value(1)

        if event.description() == "llegada":
            if i == 0:
                trans.add(State(1, j), 0.7 * self.lam)
            if j == 0:
                trans.add(State(i, 1), 0.3 * self.lam)

        elif event.description() == "servicio":
            if i == 1:
                trans.add(State(0, j), self.mu)

        elif event.description() == "transicion_j":
            if j == 1:
                trans.add(State(i, 2), self.alpha)
            elif j == 2:
                trans.add(State(i, 0), self.alpha)

        return trans



modelo = autos(lam=3.0, mu=2.0, alpha=5.0)
modelo.generate(key=lambda s: (s.get_value(0)))

modelo.print_states()

Q = modelo.get_generator_matrix()
print(Q)





            
        
