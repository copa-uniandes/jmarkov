import numpy as np

class mg1():
    """
    Implements an M/G/1 queue and computes different metrics 
    
    The M/G/1 queue has exponential interarrival times, with rate arr_rate,
    service times that follow any probability distribution, with ser_mean and ser_var,
    and 1 server.

    The class builds a birth-death chain that models this queue and computes measures
    of performance such as mean number of entities in the system, in queue, in service, 
    and the mean time in the system, in queue, and in service, as well, as utilization 
    of the server.
    """
    # arrival rate
    arr_rate:np.float64

    # service mean
    ser_mean:np.float64

    # service variance
    ser_var:np.float64

    # initializer 
    def __init__(self, arr_rate:np.float64, ser_mean:np.float64, ser_var:np.float64):
        """
        Creates an M/G/1 queue with 1 server, arr_rate arrival rate, ser_rate service rate and ser_var service variance
        """
        self.arr_rate = arr_rate
        self.ser_mean = ser_mean
        self.ser_var = ser_var

    def utilization(self)->np.float64:
        """
        Computes the mean server utilization
        
        The mean server utilization is computed as the ratio between the arrival rate
        and the service rate times the number of servers
        """
        rho = self.arr_rate/(1/self.ser_mean)
        return rho

    def mean_number_entities_queue(self)-> np.float64:
        """
        Computes the mean number of entities in queue in steady state
        """
        
        if self.is_stable:
            rho = self.utilization()
            return (((self.arr_rate**2)*(self.ser_var**2)) + (rho**2))/(2*(1-rho))
        else:
            print('Unstable queue')
            return 0
        
    def mean_number_entities_service(self)-> np.float64:
        """
        Computes the mean number of entities in service in steady state
        """
        if self.is_stable:
            return self.arr_rate/(1/self.ser_mean)
        else:
            print('Unstable queue')
            return 0
    
    def mean_number_entities(self)-> np.float64:
        """
        Computes the mean number of entities in the system in steady state
        """
        if self.is_stable:
            return self.mean_number_entities_queue() + self.mean_number_entities_service()
        else:
            print('Unstable queue')
            return 0
        
    def mean_time_system(self)-> np.float64:
        """
        Computes the mean time in the system in steady state 
        """
        if self.is_stable:
            return self.mean_number_entities()/self.arr_rate
        else:
            print('Unstable queue')
            return 0
        
    def mean_time_queue(self)->np.float64:
        """
        Computes the mean time in the queue in steady state
        """
        if self.is_stable:
            return self.mean_number_entities_queue()/self.arr_rate
        else:
            print('Unstable queue')
            return 0
    def mean_time_service(self)->np.float64:
        """
        Computes the mean time in service
        """
        if self.is_stable:
            return self.mean_number_entities_service()/self.arr_rate
        else:
            print('Unstable queue')
            return 0

    def is_stable(self)-> bool:
        """
        Returns True if the queue is stable, False otherwise
        
        This queue is stable if the arrival rate is smaller than the 
        maximum service rate
        """
        return self.arr_rate < (1/self.ser_mean) 

    

