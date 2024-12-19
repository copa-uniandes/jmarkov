import numpy as np
from jmarkov.ctbd import ctbd

class mmk():
    """
    Implements an M/M/k queue and computes steady state metrics 
    
    The M/M/k queue has exponential interarrival times, with rate arr_rate,
    exponential service times, with ser_rate,
    and k servers in parallel.

    The class builds a birth-death chain that models this queue and uses 
    the steady state probability distribution of the chain to compute
    measures of performance such as mean number of entities in the system,
    in queue, in service, and the mean time in the system, in queue, and in service.
    """
    # number of servers
    k:int

    # arrival rate
    arr_rate:np.float64

    # service rate
    ser_rate:np.float64

    # initializer 
    def __init__(self, k:int, arr_rate:np.float64, ser_rate:np.float64):
        """
        Creates an M/M/k queue with k servers, arr_rate arrival rate and ser_rate service rate
        """
        self.k=k
        self.arr_rate = arr_rate
        self.ser_rate = ser_rate

    def mean_number_entities(self)-> np.float64:
        """
        Computes the mean number of entities in the system in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if self.is_stable():
            birth = np.ones(self.k)*self.arr_rate
            death = np.arange(1,self.k+1)*self.ser_rate
            bd = ctbd(birth, death)
            n = 100
            probs = bd.steady_state(n)
            mean_num = 0 
            for i in range(n):
                mean_num += probs[i]*i

            return mean_num
        else:
            print('Unstable queue')
            return 0


    def mean_number_entities_queue(self)-> np.float64:
        """
        Computes the mean number of entities in queue in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in queue in steady state
        """
        if self.is_stable():
            birth = np.ones(self.k)*self.arr_rate
            death = np.arange(1,self.k+1)*self.ser_rate
            bd = ctbd(birth, death)
            n = 100
            probs = bd.steady_state(n)
            mean_num = 0 
            for i in range(self.k, n):
                mean_num += probs[i]*(i-self.k)

            return mean_num
        else:
            print('Unstable queue')
            return 0

    def mean_number_entities_service(self)-> np.float64:
        """
        Computes the mean number of entities in service in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if self.is_stable():
            return self.arr_rate/self.ser_rate
        else:
            print('Unstable queue')
            return 0
        
    def mean_time_system(self)-> np.float64:
        """
        Computes the mean time in the system in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the system in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        if self.is_stable():
            L = self.mean_number_entities()
            return L/self.arr_rate
        else:
            print('Unstable queue')
            return 0
    
    def mean_time_queue(self)-> np.float64:
        """
        Computes the mean time in the queue in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the queue in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        if self.is_stable():
            Lq = self.mean_number_entities_queue()
            return Lq/self.arr_rate
        else:
            print('Unstable queue')
            return 0
    
    def mean_time_service(self)-> np.float64:
        """
        Computes the mean time in service
        
        A simple relation is used to obtain the mean service time
        """
        if self.is_stable():
            return 1/self.ser_rate
        else:
            print('Unstable queue')
            return 0
    


    def is_stable(self)-> bool:
        """
        Returns True if the queue is stable, False otherwise
        
        This queue is stable if the arrival rate is smaller than the 
        maximum service rate
        """
        return self.arr_rate < self.k*self.ser_rate 
        

    
