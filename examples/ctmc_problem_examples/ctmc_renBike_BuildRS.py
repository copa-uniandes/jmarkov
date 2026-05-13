from jmarkov.markovprocess import MarkovProcess
from jmarkov.state import State
from jmarkov.event import Event
from jmarkov.transition import TransitionsSet


class rentBike(MarkovProcess):

    def __init__(self, tasallegada,tasasalida,states_list, events_list):
        self.tasallegada=tasallegada
        self.tasasalida=tasasalida
        self.states_list=states_list
        self.events_list=events_list

        super().__init__(
            states_list=self.states_list,
            events_list=self.events_list
        )
    
    def active_transitions(self, state, event):
        trans=TransitionsSet()
        i=state.get_value(0)
        
        if event.description()=="llegada":
            if i<6:
                trans.add(State(i+1),self.tasallegada)
        elif event.description()=="salida":
            if i>0:
                trans.add(State(i-1),self.tasasalida*i)
        
        return trans
    
modelo=rentBike(5,9,[State(0)],[Event(description="llegada"),Event(description="salida")])
modelo.generate()
print(modelo.get_generator_matrix())
modelo.print_states()


        
        