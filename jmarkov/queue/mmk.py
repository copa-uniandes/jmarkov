import numpy as np
from jmarkov.ctbd import ctbd

class mmk():

    # number of servers
    k:int

    # arrival rate
    arr_rate:np.float64

    # service rate
    ser_rate:np.float64

    # initializer 
    def __init__(self, k:int, arr_rate:np.float64, ser_rate:np.float64):
        self.k=k
        self.arr_rate = arr_rate
        self.ser_rate = ser_rate

    def mean_number_entities(self)-> np.float64:
        birth = np.ones(self.k)*self.arr_rate
        death = np.arange(1,self.k+1)*self.ser_rate
        bd = ctbd(birth, death)
        n = 100
        probs = bd.steady_state(n)
        mean_num = 0 
        for i in range(n):
            mean_num += probs[i]*i

        return mean_num

    def mean_number_entities_queue(self)-> np.float64:
        birth = np.ones(self.k)*self.arr_rate
        death = np.arange(1,self.k+1)*self.ser_rate
        bd = ctbd(birth, death)
        n = 100
        probs = bd.steady_state(n)
        mean_num = 0 
        for i in range(self.k, n):
            mean_num += probs[i]*(i-self.k)

        return mean_num


        

    
