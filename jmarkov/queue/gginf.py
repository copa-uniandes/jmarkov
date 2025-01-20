import numpy as np

class gginf():
    """
    Implements an G/G/Inf queue and computes several performance metrics 
    
    The G/G/Inf queue has interarrival and service times that follow any probability 
    distribution, with arr_rate and ser_rate (respectively), and the number of servers, 
    the system capacity and the population are infinite.

    The class uses the fact that the infinite number of servers simplifies may of the 
    metrics to obtain closed-form results for many average performance metrics.
    """
    # arrival rate
    arr_rate:np.float64

    # service rate
    ser_rate:np.float64

    # initializer 
    def __init__(self, arr_rate:np.float64, ser_rate:np.float64):
        """
        Creates an G/G/Inf queue with infinite servers, arr_rate arrival rate and ser_rate service rate 
        """
        self.arr_rate = arr_rate
        self.ser_rate = ser_rate

    def utilization(self)->np.float64:
        """
        Computes the mean server utilization
        
        As the queue has infinite servers, the utilization is always 0
        """
        return 0

    def mean_number_entities_queue(self)-> np.float64:
        """
        Computes the mean number of entities in queue in steady state

        As the queue has infinite servers, there is never a line in queue
        """
        return 0
        
    def mean_number_entities_service(self)-> np.float64:
        """
        Computes the mean number of entities in service in steady state
        """
        return self.arr_rate/self.ser_rate
    
    def mean_number_entities(self)-> np.float64:
        """
        Computes the mean number of entities in the system in steady state
        """
        return self.mean_number_entities_queue() + self.mean_number_entities_service()
        
    def mean_time_system(self)-> np.float64:
        """
        Computes the mean time in the system in steady state 
        """
        return self.mean_time_service()+self.mean_time_queue()
        
    def mean_time_queue(self)->np.float64:
        """
        Computes the mean time in the queue in steady state
        """
        return 0
    def mean_time_service(self)->np.float64:
        """
        Computes the mean time in service
        """
        return 1/self.ser_rate

    def is_stable(self)-> bool:
        """
        Returns True if the queue is stable, False otherwise
        
        This queue is always stable
        """
        return True

    

