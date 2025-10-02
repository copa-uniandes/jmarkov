import numpy as np
import math
from jmarkov.phase.ctph import ctph

class moments_ctph2():
    """
    Implements a PH/PH/1 queue and computes steady state metrics 
    
    The PH/PH/1 queue has phase type interarrival times, with parameters (alpha,T) of size ma,
    phase type service times, with parameters (beta,S) of size ms, 
    and 1 server.

    The class builds a quasi-birth-death chain that models this queue and uses 
    the steady state probability distribution of the chain to compute
    measures of performance such as mean number of entities in the system,
    in queue, in service, and the mean time in the system, in queue, and in service.
    """

    # first moment
    m1:float

    # second moment
    m2:float

    # phase-type adjusted to have moments m1 and m2
    PH:ctph

    
    # initializer 
    def __init__(self, m1:float, m2:float):
        """
        Creates a PH/PH/1 queue with 1 server, IAT inter-arrival times and ST service times
        """
        self.m1=m1
        self.m2=m2
       
        if m1 < 0 or m2 < 0:
            raise ValueError(f'The moments m1 ({m1}) and m2({m2}) must be positive')   
       

    def get_ph(self, verbose=False)->ctph:
        """
        Obtains the 2-phase PH representation that has the 2 moments requested
        """
        cx2 = self.m2/(self.m1*self.m1) - 1
        print(cx2)
        if cx2 < 1:
            k = math.ceil(1/cx2)
            phi = k/self.m1
            alpha = np.zeros(k)
            alpha[0] = 1
            T = np.zeros((k,k))
            for i in range(k-1):
                T[i,i] = -phi
                T[i,i+1] = phi
            T[k-1,k-1] = -phi
            self.PH = ctph(alpha,T)
        else:
            # fix 3rd moment
            m12 = self.m1*self.m1
            m13 = m12*self.m1
            #m3 = 1.5*m13*(1+cx2)*(1+cx2)
            #m3=6*m13*cx2
            m3=6*m13*cx2*2
                
            d = 2*self.m1*self.m1 - self.m2
            c = 3*self.m2*self.m2-2*self.m1*m3
            b = 3*self.m1*self.m2-m3
            a= b*b-6*c*d

            print(a,b,c,d)
            
            lda1=lda2=p=0
            if abs(c)>1e-6:
                if c > 0:
                    print('c>0')
                    sqa=math.sqrt(a)
                    p =(-b+6*self.m1*d+sqa)/( b+sqa )
                    lda1=(b-sqa)/c
                    lda2=(b+sqa)/c
                else:
                    print('c<0')
                    sqa=math.sqrt(a)
                    p = (b-6*self.m1*d+sqa)/( -b+sqa )
                    lda1=(b+sqa)/c
                    lda2=(b-sqa)/c
            else:
                print("Adjusted to exponential")
                p=0
                lda2=1/self.m1
            print(p, lda1, lda2)
            alpha = np.array([p, 1-p])    
            T = np.array([
                [-lda1, 0],
                [0, -lda2]
            ])
            self.PH = ctph(alpha,T)
        return self.PH