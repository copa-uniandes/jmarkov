import math

import numpy as np
from numpy import ndarray


# Precision for computations
epsilon:float = 1.0E-10

# Gamma function coefficients 
cof:list = [76.18009172947146, -86.50532032941677,    24.01409824083091,
		   -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
# Log factorial coefficients
a:list = [0.0] * 101

def exp_unif(A: ndarray, times: list, leftVec: ndarray, rightVec: ndarray, truncate: int=1000) -> ndarray:
    """
    Computes leftVec * exp(A x) * rightVec, for all values x in times, using the uniformization algorithm
    A: Matrix
    times: evaluation points
    leftVec: Vector
    rightVec: Vector
    truncate: upper bound for iterations
    returns leftVec * exp(A x) * rightVec
    """

    #List<Double> Avec = new ArrayList<Double>();
    # vector of ak terms
    avec = list()

    dif:float = 0.0
    lda_max = compute_lambda_max(A)
    
    #Matrix matP = getNormalized(A, lda_max);

    matP = (A.copy()*(1.0/lda_max)) + np.eye(A.shape[0])

    #n:int = times.shape[0]
    n:int = len(times)

    max_iterations: int = max(200, int(4 * lda_max * times[n - 1]))
    # TODO: change for a stochastic P 
    #m: float = 0
    M1 = rightVec.copy()
    
    #  Note: M = L (I-P)^-1 R
    temp = -matP.copy() + np.eye(matP.shape[0])
    #  M1 = (I-P)^-1 * rightVec
    #IterativeSolver solver = new BiCG(M1);
    #try {
    #    solver.solve(temp,rightVec, M1);
    #} catch (IterativeSolverNotConvergedException e) {
    #    e.printStackTrace();
    #}
    M1 = np.linalg.solve(temp, rightVec)

    m = leftVec.dot(M1)[0]
    #ak:float = 0
    ak = leftVec.dot(rightVec)[0]
    avec.append(ak)

    sumA:float = ak
    k: int = 0
    V: ndarray = np.zeros(leftVec.shape)
    
    V = matP@rightVec
    while True: 
        k+=1
        ak = leftVec.dot(V)[0] # Ak = leftMat *(P^k)*RightMat
        avec.append(ak)
        sumA += ak
        V = matP@V # // Vk = P * RightMat

        dif_mat = sumA
        dif_mat = dif_mat - m
        dif = np.abs(dif_mat)
        #print(f"dif: {dif}")

        if (dif < epsilon) or (k >= max_iterations) or (k >= truncate):
            break
 

    maxK:int  = k
    #double As[] = new double[maxK + 1];
    #for (k = 0; k <= maxK; k++) {
    #    As[k] = Avec.get(k).doubleValue();
    #}
    #As = np.array(avec)
    As = avec.copy()
    #double[] result = new double[n];
    result = np.zeros(n)
    for i in range(n): #(int i = 0; i < n; i++) {
        ldaX = lda_max * times[i]
        pk = np.exp(-ldaX)

        if (pk < np.finfo(float).tiny): 
            #print("pk menor que np.finfo(float).tiny")
            result[i] = resultFromMedian(As, ldaX)
        else:
            sumPk = pk
            #print(f"As[0]: {As[0]}, result[i]: {result[i]}")
            result[i] = As[0]
            result[i] *= pk

            for k in range(1, maxK + 1): 
                pk = pk * (ldaX / k)
                sumPk += pk
                #print(f"As[{k}]: {As[k]}, result[i]: {result[i]}, pk: {pk}")
                result[i] = result[i] + pk * As[k]
    
    return result



def compute_lambda_max(A: ndarray) -> float:
    """Computes the maximum value of the negative diagonal elements of the generator matrix A."""
    temp = -np.diag(A)
    lambda_max = np.max(temp)

    return lambda_max

def resultFromMedian(A: list, ldaX: float) -> float:
    """
    Computes the value of exp(A[i] x) when it is to small
    @param A double
    @param ldaX
    @return exp(A[i] x)
    """
    
    if (ldaX == 0):
        return A[0]
    median:int = math.floor(ldaX)
    p_plus = math.exp(-ldaX + median * math.log(ldaX) - lnFactorial(median))
    p_minus = p_plus
    sumPk = p_plus
    result = A[median]* p_plus if median < len(A) else 0

    k_minus = median
    k_plus = median
    maxK = max(median, len(A) - median + 1)
    for k in range(1, maxK):
        if (sumPk >= 0.999):
            break
        k_minus -= 1
        k_plus += 1
        p_plus = p_plus * (ldaX / k_plus)
        p_minus = p_minus * (k_minus + 1) / ldaX
        sumPk += p_plus + p_minus
        if (k_plus < len(A)):
            result = result + p_plus * A[k_plus]
        if ((k_minus < len(A)) and (k_minus >= 0)):
            result = result + p_minus * A[k_minus]
    
    return result
    
    
def lnFactorial(n: int ) -> float:    
    """Computes the log of Factorial function
	 * @param n
	 * @return ln (n!)
	 """
    if n<0:
        raise ArithmeticError("Negative factorial")
    if (n<=1): return 0.0
    if (n<=32): return a[n] if a[n]!=0 else math.log(math.factorial(n))
    if (n<=100): 
        if (a[n]==0):
            a[n]=lnGamma(n+1.0)
        return a[n] 
    else: 
        return lnGamma(n+1.0)


def lnGamma(xx: float) -> float:
    """
    Computes the log of gamma function.
    @param xx value
    @return lnGamma(xx)
    """
    y=x=xx
    tmp=x+5.5
    tmp -= (x+0.5)*math.log(tmp)
    ser = 1.000000000190015
    for j in range(6):  
        y+=1
        ser += cof[j]/y
    
    return (float)(- tmp + math.log(2.506628274631005*ser/x))
