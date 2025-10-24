from config import *




# Cantilever Beam
def analy_cb(x):
    return - (q / (E * I)) * (x**4 / 24 - x**3 / 6 + x**2 / 4)



# Fully Restrained Beam
def analy_frb(x):
    return (-q / (24 * E * I)) * (x**4 - 2 * L * x**3 + L**2 * x**2)



# Fully Restrained Beam with Mid-Span Point Load
def analy_frb_mspl(x):
    y = np.zeros_like(x)
    half_L = L / 2
    mask1 = x <= half_L
    mask2 = x > half_L

    y[mask1] = (q / (48 * E*I)) * (3 * L * x[mask1]**2 - 4 * x[mask1]**3)
    y[mask2] = (q / (48 * E*I)) * (3 * L * (L - x[mask2])**2 - 4 * (L - x[mask2])**3)
    return y