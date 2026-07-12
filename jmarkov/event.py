

class Event:
    """
    Represents a single event in a continuous time markov chain.
    """
    def __init__(self, description: str = ""):
        """
        description : str
        A string that describes the name of the event
        """
        self._index = -1
        self._set = None
        self._description: str = description

    def set_set(self, event_set):
        self._set = event_set

    def get_index(self):
        """Returns the index assigned by the EventSet"""
        return self._index

    def set_index(self, num):
        """Sets the index of the event."""
        self._index = num

    # Comparison
    def __lt__(self, other):
        return self._index < other.get_index()

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self._index == other.get_index()

    def label(self):
        """Short label that shows the string event plus the index"""
        return f"Event {self._index}"
    
    def description(self):
        """Returns the description provided at the construction of the event"""
        return self._description

    def __str__(self):
        return self.label()


# Events SET:Represents the complete set of events of the model.
class EventsSet:
    """
    Ordered collection of Event objects that represents all the events in the Markov model.
    """
    def __init__(self, event_list=None):
        """
        event_list : list of Event objects.
        """
        self._events = []

        if event_list:
            for e in event_list:
                self.add(e)

    def add(self,event):
        """Registers an Event in the set, assigns the index, and keeps the list order."""
        event.set_set(self)
        event.set_index(len(self._events))
        self._events.append(event)
        self._events.sort()  # mantains index order
        
    def contains(self, event):
        """True if the Event belongs to this EventsSet."""
        return event in self._events

    def size(self):
        """Returns the number of Events in the set."""
        return len(self._events)

    def to_event_array(self):
        """Returns a list of all events."""
        return list(self._events)

    def __iter__(self):
        return iter(self._events)

    def __str__(self):
        return str(self._events)