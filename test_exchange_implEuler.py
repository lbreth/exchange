from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from micromagnetictestcases.macrospin.analytic_solution import macrospin_analytic_solution

# Material parameters
A = 1.3e-11 # exchange constant (J/m) for Permalloy from http://www.ctcms.nist.gov/~rdm/std4/spec4.html
Ms = 8.6e5  # saturation magnetisation (A/m)
alpha = 0.02  # Gilbert damping
gamma = 2.211e5  # gyromagnetic ratio

# External magentic field.
B = 0.1  # (T)
mu0 = 4 * np.pi * 1e-7  # vacuum permeability
H = B / mu0
# meaningful time period is of order of nano seconds
dt = 1e-12
t_array = np.arange(0, 5e-9, dt)

C = 2*gamma*A/(mu0*Ms)

############
# Simulation
############

# mesh parameters
d = 50e-9
thickness = 10e-9
nx = ny = 10
nz = 1

# create mesh
p1 = Point(0, 0, 0)
p2 = Point(d, d, thickness)
mesh = BoxMesh(p1, p2, nx, ny, nz)

# define function space for magnetization
M = VectorFunctionSpace(mesh, "CG", 1, dim=3)

# define initial M and normalise
m = Constant((1, 0, 0))
H_z = H * Constant((0, 0, 1))

# define initial value
u_n = interpolate(m, M)


# define variational problem
u = Function(M)
# u = TrialFunction(M)
v = TestFunction(M)



F = dot(u - u_n, v)*dx + dt*gamma*dot(cross(u, H_z), v)*dx - alpha*dot(cross(u, u_n), v)*dx - C*dt*inner(grad(cross(v,u)),grad(u))*dx
#a, L = lhs(F), rhs(F)

# time stepping
# u = Function(M)
t = 0
mx_simulation = np.zeros(t_array.shape)

for i, t in enumerate(t_array):

    mx_simulation[i] = u((0,0,0))[0]
    t += dt
    # solve(a == L, u)
    solve(F == 0, u)

    # plot solution
    # plot(u)

    # update previous solution
    u_n.assign(u)


# Hold plot
interactive()
