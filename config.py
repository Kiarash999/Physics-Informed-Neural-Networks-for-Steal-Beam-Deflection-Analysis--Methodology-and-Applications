import numpy as np
import deepxde as dde
import deepxde.backend.tensorflow_compat_v1 as tf
import matplotlib.pyplot as plt



# porperties
L = 1.0
E = 200e9
I = 1e-6
q = 10000.0
c = q / (E * I)
sigma =0.01



# configuration
soft_const = True