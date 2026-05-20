
from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class Corrosion(MarkovProcess):
    """
    Example with 2 state variables where x(t) = Is the pump corrosion level, 
    y(t) = Is the level of impact on the environment. Where Sx = {0, 1, 2,..., 5},
    S(y) = {low, high}
    """

    lam_alto:float
    lam_bajo:float
    lam_alto_bajo:float
    lam_bajo_alto:float
    max:int
    min:int

    def __init__(self, lam_alto, lam_bajo, lam_alto_bajo, lam_bajo_alto,max,min):
        self.lam_alto      = lam_alto        
        self.lam_bajo      = lam_bajo        
        self.lam_alto_bajo = lam_alto_bajo   
        self.lam_bajo_alto = lam_bajo_alto   
        self.max = max
        self.min = min

        
        super().__init__(
            states_list=[State(1, "Alto")],
            events_list=[
                Event(description="Corrosion_Sube"),
                Event(description="Corrosion_Baja"),
                Event(description="Ambiente_Baja"),
                Event(description="Ambiente_Sube"),
            ]
        )

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        trans = TransitionsSet()
        i = state.get_value(0)   
        j = state.get_value(1)   

        if event.description() == "Corrosion_Sube":
            if i < self.max:
                trans.add(State(i + 1, j), self.lam_alto)

        elif event.description() == "Corrosion_Baja":
            if i > self.min:
                trans.add(State(i - 1, j), self.lam_bajo)

        elif event.description() == "Ambiente_Baja":
            if j == "Alto":
                trans.add(State(i, "Bajo"), self.lam_alto_bajo)

        elif event.description() == "Ambiente_Sube":
            if j == "Bajo":
                trans.add(State(i, "Alto"), self.lam_bajo_alto)

        return trans


modelo = Corrosion(
        lam_alto      = 0.1,   
        lam_bajo      = 0.2,   
        lam_alto_bajo = 0.3,   
        lam_bajo_alto = 0.4,   
        max = 5,
        min = 1
    )

    

modelo.generate(key=lambda s: (s.get_value(0)))

modelo.print_states()

Q = modelo.get_generator_matrix()
print(Q)

    
print("\nVerificacion - suma de filas de Q:")
for i, row in enumerate(Q):
    print(f"  fila {i}: {sum(row):+.6f}")
