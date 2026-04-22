
from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class Corrosion(MarkovProcess):

    def __init__(self, lam_alto, lam_bajo, lam_alto_bajo, lam_bajo_alto,max,min):
        self.lam_alto      = lam_alto        
        self.lam_bajo      = lam_bajo        
        self.lam_alto_bajo = lam_alto_bajo   
        self.lam_bajo_alto = lam_bajo_alto   
        self.max = max
        self.min = min

        
        super().__init__(
            states_list=[State(1, "alto")],
            events_list=[
                Event(description="corrosion_sube"),
                Event(description="corrosion_baja"),
                Event(description="ambiente_baja"),
                Event(description="ambiente_sube"),
            ]
        )

    def active_transitions(self, state: State, event: Event) -> TransitionsSet:
        trans = TransitionsSet()
        i = state.get_value(0)   
        j = state.get_value(1)   

        if event.description() == "corrosion_sube":
            if i < self.max:
                trans.add(State(i + 1, j), self.lam_alto)

        elif event.description() == "corrosion_baja":
            if i > self.min:
                trans.add(State(i - 1, j), self.lam_bajo)

        elif event.description() == "ambiente_baja":
            if j == "alto":
                trans.add(State(i, "bajo"), self.lam_alto_bajo)

        elif event.description() == "ambiente_sube":
            if j == "bajo":
                trans.add(State(i, "alto"), self.lam_bajo_alto)

        return trans


modelo = Corrosion(
        lam_alto      = 0.1,   
        lam_bajo      = 0.2,   
        lam_alto_bajo = 0.3,   
        lam_bajo_alto = 0.4,   
        max = 5,
        min = 1
    )

    
orden_ambiente = {"alto": 0, "bajo": 1}
modelo.generate(key=lambda s: (s.get_value(0), orden_ambiente[s.get_value(1)]))

modelo.print_states()

Q = modelo.get_generator_matrix()
print(Q)

    
print("\nVerificacion - suma de filas de Q:")
for i, row in enumerate(Q):
    print(f"  fila {i}: {sum(row):+.6f}")







    
    