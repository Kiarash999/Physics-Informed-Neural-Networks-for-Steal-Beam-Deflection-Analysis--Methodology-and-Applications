from config import *
from geom import geom





def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], L)

def ddy(x, y):
    return dde.grad.hessian(y, x)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

def boundary_second_derivative(x, y, _):
    return ddy(x, y)

def boundary_third_derivative(x, y, _):
    return dddy(x, y)




bcD_l = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bcN_l = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)

bcO_r2 = dde.icbc.OperatorBC(geom, boundary_second_derivative, boundary_r)
bcO_r3 = dde.icbc.OperatorBC(geom, boundary_third_derivative, boundary_r)

bcD_r = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_r)
bcN_r = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_r)


# bcs for (Cantilever Beam)
bcs_c = [bcD_l, bcN_l, bcO_r2, bcO_r3] if soft_const == True else []


# bcs for (Fully Restrained Beam) & (Fully Restrained Beam with Mid-Span Point Load)
bcs_fr = [bcD_l,  bcN_l, bcD_r, bcN_r] if soft_const == True else []
