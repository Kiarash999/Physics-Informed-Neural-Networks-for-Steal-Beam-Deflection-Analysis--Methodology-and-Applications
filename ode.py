from config import *




# Euler–Bernoulli ODE
def ode(x ,y):
    dy_xx = dde.grad.hessian(y, x)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + c




# Euler–Bernoulli ODE (Fully Restrained Beam with Mid-Span Point Load)
def ode_mspl(x, y):
    d2y = dde.grad.hessian(y, x, i=0)
    d4y = dde.grad.hessian(d2y, x, i=0)
    return E*I * d4y - q / (sigma * np.sqrt(2 * np.pi)) * tf.exp(-((x - L / 2) ** 2) / (2 * sigma ** 2))