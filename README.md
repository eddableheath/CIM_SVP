# Quantum Discrete Hybrid Monte Carlo

Using continuous-time discrete space quantum walks over planar lattice graphs in concert with a Metropolis filter to solve SVP.

Fairly basic free particle walk on integer lattice given by:

U = exp(iHt)

where t is the propagation time and H is the Hamiltonian:

H = -gL

where g is the hopping amplitude and L is the graph Laplacian,

L = A - D

where A is the adjacency matrix for the graph and D_{jj} = deg(j).

Given a starting state of |j> we apply U and then measure to give a new proposal state. This is then compared to the previous state using a Metropolis filter.

This scales poorly if you want to ensure the existence of the shortest vector with the scope of the integer plane, however there are several proposed remedies for this:
* Explore a smaller area, expand if the shortest vector not found.
  * This area could be defined around the starting state and move based on the new starting state.
* Explore two dimensions at a time.