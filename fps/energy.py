'''
Functions to compute the contribution of a single particle to overall
potential energy with in the run() function from sampling.py

Copyright (C) 2026 Miriam Derla

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>
'''

import numpy as np
import numba as nb

from .indexing import __flatten_multiindex, __inflate_flattened_index



@nb.njit
def u(r, E, sigma):
	'''
	pair potential for penetrable spheres or disks given a radius

	Args:
		r:
			the distance of the particle pair in question
		E:
			the size of the energy step
		sigma:
			the distance at which the energy step happens
	'''
	return E if r < sigma else (E/2 if r == sigma else 0)



@nb.njit
def single_particle_potential_energy(
	# particle information
	# NOTE: we pass x_i separately because we want to move ...
	#		... it relative to x[i] for the metropolis-hastings- ...
	#		... sampling move
	i, x_i, x,
	# which group and cell are we in
	igi, g,
	# dimensionality and cell list
	spatial_dimension_count, divisions,
	cell_list, nearest_neighbour_count,
	lattice_vectors,
	# physical parameters
	overlap_energy,
	sigma
	):
	'''
	Computes the contribution of a single particle to
	potential energy; to be more precise, if i is the
	index of the particle under consideration, then
	we compute the sum of all u(r_ij) over all j != i
	'''
	contribution_to_potential_energy = 0
	# grabbing the positions of all particles in the current and all next neighbouring ...
	# ... cells with a running multiindex-shift n (as in "neighbours"). Note that we first ...
	# ... need to loop over n_k = -1, 0, 1 to then compute from this the shift in cell ...
	# ... index n_0 + n_1 * M + n_2 * M^2 + ... + n_{d-1} * M^(d-1)
	for flattened_index_shift_to_nearest_neighbour in range(nearest_neighbour_count):
		n = __inflate_flattened_index(
			flattened_index_shift_to_nearest_neighbour,
			spatial_dimension_count, 3
		) - 1 # NOTE the minus one maps 0 -> -1 and 1 -> 0 and 2 -> 1, exactly what we want
		nearest_neighbour_cell_index = __flatten_multiindex(
			(3 * igi + g + n),
			spatial_dimension_count,
			divisions
		)
		for j in cell_list[nearest_neighbour_cell_index]:
			# no self-interaction
			if j == i:
				continue
			# TODO 	do not use the watering can like this, be selective and
			#		only consider images which could even be relevant
			for R in lattice_vectors:
				x_j = x[j]
				r_ij = np.linalg.norm(x_i - x_j + R)
				contribution_to_potential_energy += u(r_ij, overlap_energy, sigma)
	return contribution_to_potential_energy