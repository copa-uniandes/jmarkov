#!/usr/bin/env python
import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.fit.moments_ctph2 import moments_ctph2
from jmarkov.phase.ctph import ctph

m1 = 1
cx2 = 2
m2 = m1*m1*(cx2+1)

fitter = moments_ctph2(m1,m2)
PH = fitter.get_ph()
print(PH.alpha)
print(PH.T)