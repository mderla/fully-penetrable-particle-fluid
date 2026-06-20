'''
Implementing a Metropolis-Hastings sampling of fully penetrable
spheres with numba

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

from random import shuffle

from .energy import single_particle_potential_energy
from .indexing import __flatten_multiindex, __inflate_flattened_index, __repopulate_cell_list
from .geometry import B



@nb.njit(parallel=True)
def run(
	overlap_energy, kB_times_temperature,
	packing_density, particle_number,
	spatial_dimension_count=3, # NOTE might want to vary in case we find a second-order transition
	simulation_box_size=1,
	step_size_in_particle_diameters=0.25,
	iteration_count=1000,
	x_init=None):
	'''
	Executing the metropolis hastings sampling of
	the penetrable sphere system
		
	Args:
		overlap_energy:
			the energy necessary to make to particles entirely coincide
		kB_times_teperature:
			Boltzmann-constant times temperature (the other characteristic
			energy scale besides overlap energy)
		packing_density:
			the fraction of the volume that would be occupied by the particles
			if none of them overlapped
		particle_number:
			the number of particles in the system
		spatial_dimension_count:
			the number of spatial dimensions that particles move in
		simulation_box_size:
			the sidelength of the simulation box; this parameter effectively
			defines the used units, since all other length-scales are derived
			from it
		step_size_in_particle_diameters:
			how many particle diameters the standard deviation of the (normal
			distributed) particle moves is
		iteration_count:
			one iteration consits of sampling a move for every particle, this
			parameter is the number of iterations the function will perform
		x_init:
			if set to something other than None, it will be used as the initial
			state. Intended use is dumping the state to a .xyz file, since it
			allows us to do a limited number of moves, pass that result back to
			python, use file-IO to dump, then resume the simulation at the last
			configuration it was in by passing the latter to x_init
	
	Returns:
		the last state of the system. Intended to be used for resuming the
		sampling procedure there if necessary
	'''
	# lattice vectors whose corresponding periodic images ...
	# ... are relevant to potential energy
	lattice_vectors = [
		__inflate_flattened_index(
			lattice_vector_flattened_index,
			spatial_dimension_count, 3) - 1
		for lattice_vector_flattened_index
		in range(3**spatial_dimension_count)
	]
	# computing the particle diameter corresponding to the ...
	# ... sought packing density
	volume = simulation_box_size**spatial_dimension_count
	number_density = particle_number / volume
	sigma = 2 * (
		packing_density / (
			B(spatial_dimension_count) * number_density)
			)**(1/spatial_dimension_count)
	# computing the step size
	step_size = step_size_in_particle_diameters * sigma
	# randomly placed particles
	x = (
		np.random.rand(particle_number, spatial_dimension_count) * simulation_box_size
		if x_init is None
		else x_init.copy()
	)
	# cell list for energy computation, where the number of divisions is ...
	# ... chosen to be multiples of three, in order to update cells later in ...
	# ... 3^d groups (hence the factors of 3 and 1/3 sandwiching the floor-operation)
	third_of_divisions = int(np.floor((1/3)*simulation_box_size / sigma)) # NOTE: used a lot later to inflate flattened indices
	divisions = int(3 * third_of_divisions)
	cell_list_size = divisions**spatial_dimension_count
	update_group_count = 3**spatial_dimension_count
	nearest_neighbour_count = 3**spatial_dimension_count # NOTE: may be different if we later choose larger nearest-neighbours
	cells_per_update_group = int(cell_list_size / update_group_count)
	
	# functions need a fixed return type, so we cannot recursively construct ...
	# ... the cell list without some overloading magic. Hence I will just use ...
	# ... a flattened index
	cell_list = nb.typed.List([
		# the number of divisions per dimension
		nb.typed.List.empty_list(nb.types.int64)
		for _ in range(cell_list_size)
	])
	# tracking acceptance rate
	single_particle_acceptance_rates = np.empty(iteration_count)
	for iteration in range(iteration_count):
		# executing the sampling moves where we update ...
		# ... non-interacting cells in parallel, i.e. one of ...
		# ... 3^d groups of cells which are all out of ...
		# ... interaction range of each other; cells, separated ...
		# ... by three in any direction instead of one, are ...
		# ... updated, i.e. i_k = 3 * igi_k + g_k, where g=(g_0,...g_{d-1}) is the ...
		# ... index of the group, as represented by a virtual 3x3x...x3 ...
		# ... configuration of cells and igi_k is the cell index
		for flattened_group_index in range(update_group_count):
			# every iteration, we count the number of accepted moves ...
			# ... for each update group
			accepted_moves_in_g = 0
			denied_moves_in_g = 0
			# inflating the update group index to (g_0,...,g_{d-1})
			g = __inflate_flattened_index(
				flattened_group_index,
				spatial_dimension_count,
				# will always be three along any given dimension: ...
				# ... the cell itself and its left and right neighbours
				3 
			)
			# we want the cell list to reflect moves done while ...
			# ... updating the previous group
			__repopulate_cell_list(
				x, cell_list,
				divisions, simulation_box_size
			)
			# we now update all cells of group g in parallel
			for intra_group_index in nb.prange(cells_per_update_group):
				# figuring out the cell index
				igi = __inflate_flattened_index(
					intra_group_index,
					spatial_dimension_count,
					# since update group g has only 1/3^d the number of cells in it, so a third ...
					# ... of the divisions per dimension
					third_of_divisions
				)
				cell_index = __flatten_multiindex(
					3 * igi + g,
					spatial_dimension_count,
					divisions
				)
				# for all particles in the cell in question, we will now
				for i in cell_list[cell_index]:
					x_i = x[i]
					# computing contribution U_i of particle i to overall potential energy
					U_i = single_particle_potential_energy(
						i, x_i, x,
						# necessary for indexing, cell list magic, etc.
						igi, g,
						spatial_dimension_count, divisions,
						cell_list, nearest_neighbour_count,
						lattice_vectors,
						# physical parameters
						overlap_energy,
						sigma
					)
					# drawing a move ...
					x_i_try = (
						x_i + np.random.normal(
							0,
							step_size,
							size=spatial_dimension_count
							)) % simulation_box_size
					# ... computing how that changes potential energy ...
					U_i_try = single_particle_potential_energy(
						i, x_i_try, x,
						# necessary for indexing, cell list magic, etc.
						igi, g,
						spatial_dimension_count, divisions,
						cell_list, nearest_neighbour_count,
						lattice_vectors,
						# physical parameters
						overlap_energy,
						sigma
					)
					dU = U_i_try - U_i
					# accepting outright of it lowers energy ...
					move_i_accepted = False
					if dU < 0:
						move_i_accepted = True
					else:
						# ... otherwise draw a random number
						p_accept = min(1, np.exp(-dU/kB_times_temperature))
						if np.random.rand() < p_accept: 
							move_i_accepted = True
					# if move was accepted, alter x[i] and track acceptance rate
					if move_i_accepted:
						x[i] = x_i_try
						accepted_moves_in_g += 1
					else:
						denied_moves_in_g += 1
			# accounting for the possibility that there were no moves at all, ...
			# ... we will skip trying to make a contribution if that is the case
			if accepted_moves_in_g + denied_moves_in_g > 0:
				single_particle_acceptance_rates[iteration] += accepted_moves_in_g / (
					accepted_moves_in_g + denied_moves_in_g
				)
		# shuffling all particles to not accidentally introduce ...
		# ... effects only due to the fixed update order
		shuffle(x)
		# normalizing the sum of acceptance rates to a fraction
		single_particle_acceptance_rates[iteration] /= update_group_count
	return x, single_particle_acceptance_rates