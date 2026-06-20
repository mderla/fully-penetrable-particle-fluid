'''
Geometric functions used in run() from sampling.py

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

from math import lgamma

import numpy as np
import numba as nb



@nb.njit
def B(d):
    '''-
    Returns the hyper-volume of the unit ball
    in d dimensions
    
    Args:
		d:
			the number of spatial dimensions the (hyper-)unit
            ball exists in

    https://en.wikipedia.org/wiki/N-sphere#Volume_and_area
    '''
    gamma_of_1_plus_d_over_2 = np.exp(lgamma(1 + d/2))
    return np.pi**(d/2) / gamma_of_1_plus_d_over_2