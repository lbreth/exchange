from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from micromagnetictestcases.macrospin.analytic_solution import macrospin_analytic_solution

# Material parameters
A = 1.3e-11 # exchange constant (J/m) for Permalloy from http://www.ctcms.nist.gov/~rdm/std4/spec4.html
Ms = 8.6e5  # saturation magnetisation (A/m)
alpha = 0.02  # Gilbert damping
gamma = 2.211e5  # gyromagnetic ratio



# External magnetic field.
B = 0.1  # (T)
mu0 = 4 * np.pi * 1e-7  # vacuum permeability
H = B / mu0
# meaningful time period is of order of nano seconds
dt = 1e-11

C = (A*gamma)/(2*mu0*Ms)
t_array = np.arange(0, 5e-9, dt)

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
m_init = Constant((1, 0, 0))
H_z = H * Constant((0, 0, 1))

# define initial value
m_n = interpolate(m_init, M)


# define variational problem
m = Function(M)
v = TestFunction(M)

F = - dot((m - m_n),v)*dx - gamma*dt*0.5*dot(cross((m + m_n), H_z),v)*dx \
    + C*dt*inner(grad(cross(v, (m + m_n))), grad(m + m_n))*dx \
    + alpha*inner(cross(m_n, m), v)*dx


# time stepping
t = 0

for t in t_array:

    t += dt
    solve(F == 0, m )
   
    # plot solution
    # plot(u)
 
    # update previous solution
    m_n.assign(m)

# Hold plot
# interactive()










# Sub domain for Dirichlet boundary condition
# class DirichletBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary


# g = Constant((1, 0, 0))
# bc = DirichletBC(M, g, DirichletBoundary())

