'''
Functions to flatten multiindices and inflating flattened indices
as well as a function to populate the cell list in run() from
sampling.py

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



@nb.njit
def __inflate_flattened_index(
	flattened_index,
	spatial_dimension_count,
	indices_per_dimension):
	'''
	Computes the multiindex (i_0, i_1, ..., i_{d-1}) from the
	flattened cell index i_0 + i_1 * M + ... + i_{d-1} * M^(d-1)
		
	Args:
		flattened_index:
			the flattened index to inflate
		spatial_dimension_count:
			d from the above description
		indicies_per_dimension:
			M from the above description
		
	Returns:
		(i_0, i_1, ..., i_{d-1}) as an np.int64 array with d entries
	'''
	flattened_index_copy = int(flattened_index) # making a copy
	inflated_multiindex = np.zeros(spatial_dimension_count, dtype=np.int64) # output for later 
	for k in range(spatial_dimension_count):
		# let M = indices_per_dimension for brevity:
		# flattened_index = i_0 + i_1 * M + i_2 * M^2 + ... + i_{d-1} * M^(d-1), hence ...
		# ... flattened_index % M = i_0, making flattened_index - i_0 = i_1 * M + ..., such ...
		# ... that way may divide out M and repeat the process
		i_k = flattened_index_copy % indices_per_dimension
		inflated_multiindex[k] = i_k
		flattened_index_copy -= i_k
		# TODO dividing is nono. Can you avoid it?
		flattened_index_copy = int(flattened_index_copy / indices_per_dimension)
	return inflated_multiindex



@nb.njit
def __flatten_multiindex(multiindex, spatial_dimension_count, indices_per_dimension):
	'''
	Takes in a multiindex (i_0, i_1, ..., i_{d-1}) and flattens it into
	a cell index i_0 + i_1 * M + ... + i_{d-1} * M^(d-1). If any of the
	i_k exceeds M or falls below 0, i_k % M will be used in its stead.
		
	Args:
		multiindex:
			the (i_0, i_1, ..., i_{d-1}) to be flattened
		spatial_dimension_count:
			the number of spatial directions d implied
		indices_per_dimension:
			starting from 0, the maximum value that the i_k reach
	'''
	flattened_index = 0
	for l in range(spatial_dimension_count):
		flattened_index += int(
			# sparing user from having to watch out for in-range indices and ...
			# ... hence enable them to increment up and down freely
			(multiindex[l] % indices_per_dimension)
			* indices_per_dimension**l
		)
	return flattened_index



@nb.njit
def __repopulate_cell_list(
	x, cell_list,
	divisions, simulation_box_size):
	'''
	takes in the array of positions and fills them into the cell list
		
	Args:
		bin_indices:
			list of bin indices of size spatial_dimension_count
		cell_list:
			the cell_list
		divisions:
			the number of divisions per independent spatial direction
		simulation_box_size:
			the size of the simulation box, given in the same units as
			the particle diameter
	'''
	# retrieving particle number from x
	particle_number, spatial_dimension_count = x.shape
	cell_list_size = len(cell_list)
	# clearing all lists entries
	for i in range(cell_list_size):
		cell_list[i].clear()
	# bin indices are given by flooring
	bin_indices = np.floor(divisions * x / simulation_box_size).astype(np.int64) % divisions
	for k in range(particle_number):
		b_k = bin_indices[k]
		cell_list[__flatten_multiindex(b_k, spatial_dimension_count, divisions)].append(k)