'''
IO utility for the XYZ file format

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

from typing import Iterator
from dataclasses import dataclass
from pathlib import Path
from logging import exception

from numpy import array
from numpy import number
from numpy.typing import NDArray


@dataclass(frozen=True)
class XYZ:
    '''
    class whose properties reflect all information of an
    XYZ block. Note that the interpretation of the columns
    is up to the user

    Properties:
        particle_number:
            the number of particles in the block
        comment:
            a string containing the comment line
        labels:
            the particle labels
        xyz_data:
            the numerical XYZ-data columns

    Methods:
        is_valid:
            checks whether the XYZ-block is consistent

    Static Methods:
        file_iterator:
            returns an iterator over all XYZ-blocks in
            an XYZ-file at the passed filepath

    For XYZ file format reference, see:
        https://en.wikipedia.org/wiki/XYZ_file_format
    '''
    particle_number:int
    comment:str
    labels:list[str]
    xyz_data:NDArray

    @property
    def is_valid(self):
        '''
        Checks whether the XYZ block is valid
        '''
        label_count = len(self.labels)
        xyz_row_number = self.xyz_data.shape[0]
        return label_count == self.particle_number and xyz_row_number == self.particle_number

    @property
    def column_count(self):
        '''
        Returns the number of columns
        '''
        _, column_count = self.xyz_data.shape
        return column_count

    def __str__(self):
        '''
        The standard __str__ output is too cumbersome due to xyz_data
        '''
        label_set = set(self.labels)
        label_count = len(label_set)
        comment_len = len(self.comment)
        return (
            f'XYZ({self.particle_number} particles, ' +
            f'{self.column_count} columns, ' +
            (
                f'labels: {label_set}'
                if label_count <= 3
                else f'{label_count} distinct labels'
            ) + ', ' +
            f'comment{(
                f": '{self.comment}'"
                if comment_len <= 50
                else f" has {comment_len} characters"
                )})'
        )

    def __repr__(self):
        '''
        The standard __repr__ output is too cumbersome due to xyz_data
        '''
        return self.__str__()

    def __getitem__(self, index:int) -> tuple[str, ...]:
        if index < self.__len__() and index >= 0:
            return (self.labels[index], *self.xyz_data[index])
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        '''
        The only useful way in which the block can be considered linearly
        extended is in the number of rows, which is the number of particles
        '''
        return self.particle_number


    @staticmethod
    def open(file_path:Path) -> Iterator['XYZ']:
        '''
        Iterator that on each iteration returns a block of an
        XYZ file

        Arguments:
            file_path (Path):
                path of the XYZ-file over whose blocks should
                be iterated
        '''

        # ensure filepath is path by using Path(Path(x)) = Path(x)
        file_path = Path(file_path)

        with file_path.open('r', encoding='ascii') as file:

            # while loop always causing another readline
            while line := file.readline():

                # we ensure that the line read in by the ...
                # ... while-loop is the particle number ...
                particle_number = int(line.strip())

                # ... by reading in the comment line ...
                comment = file.readline().strip()

                # ... followed by a reliable number of lines
                labels = []
                xyz_data = []
                for _ in range(particle_number):

                    # splitting the label at the beginning of the line from the ...
                    # ... numerical information
                    particle_descriptor, *xyz_columns = file.readline().split(' ')

                    # appending them to separate lists
                    labels.append(particle_descriptor)
                    xyz_data.append([float(column) for column in xyz_columns])

                # NOTE: when this line raises a ValueError due to inhomogeneous ...
                # ... xyz_data, it is likely because particle number and line-count ...
                # ... don't match up, since that is easier to miss than inhomogeneous ...
                # ... row lengths
                xyz_data = array(xyz_data)

                yield XYZ(
                    particle_number=particle_number,
                    comment=comment,
                    labels=labels,
                    xyz_data=xyz_data
                )


    @staticmethod
    def dump(
            file_path:Path,
            *columns:tuple[NDArray|list[number]],
            labels:list[str]=None,
            comment:str=None) -> None:
        '''
        Dumps an xyz file (Wikipedia format description:
        https://en.wikipedia.org/wiki/XYZ_file_format) of
        a given configuration to the passed path. Appends
        the data if the file exists, creates a new one if
        it does not. Columns can be lists or numpy arrays up
        to rank 2 and contain the values of one property
        for all particles, e.g. all x-positions or all radii.

        Arguments:
            file_path:
                path that the data is supposed to be written to
            *columns:
                arbitrarily many arguments after file_path where
                one can pass columns that should be written to
                the file. Columns can be lists or numpy arrays up
                to rank 2 and contain the values of one property
                for all particles, e.g. all x-positions or all radii
            labels:
                particle labels that specify the particle type or
                perhaps identity.
            comment:
                XYZ files allow a comment to be passed in each
                block, that can contain anything, from messages to
                file recipients to data not captured by the numbers
                passed as columns, like simulation parameters
        Returns:
            None
        '''

        # labels cannot contain spaces since these are used as ...
        # ... delimiters in XYZ
        if labels is not None:
            if any(' ' in label for label in labels):
                raise ValueError(
                    'Labels contain whitespace, but they are used in XYZ as delimiters'
                )

        # ensure filepath is path by using Path(Path(x)) = Path(x)
        file_path = Path(file_path)

        # concatenating all columns into one numpy ndarray, where ...
        # ... lists are converted by np.array(np.array()). The ...
        # ... difference between homogeneous_columns and colomns ...
        # ... is that columns are supposed to be lists, numpy arrays ...
        # ... or numpy 2darrays
        homgeneous_columns = []
        for column in columns:

            # make sure the columns are numpy arrays
            column = array(column)

            if len(column.shape) == 1:

                # 1d arrays can be added right to the list
                homgeneous_columns.append(column)

            elif len(column.shape) == 2:
                # 2d arrays have to be split first
                proper_columns = [proper_column for proper_column in column.T]
                homgeneous_columns += proper_columns

            else:
                # wrong input
                raise ValueError('columns can only be lists or numpy 1d arrays and 2d arrays')

        # attempting to merge the columns into one array
        try:
            data = array(homgeneous_columns)
        except ValueError: # as columns_inhomogeneous:
            exception('All columns need to be of the same length!')

        # correctly writing the entry block
        particle_count, _ = data.T.shape

        with file_path.open('a', encoding='ascii') as dump_file:

            # adding particle count
            dump_file.write(f'{particle_count}\n')

            # adding particle count
            dump_file.write(f'{comment if comment is not None else ""}\n')

            # writing the configurational data of the particles to file
            dump_file.writelines(
                ' '.join([

                    # setting some generic name that indicates we ...
                    # ... did not / forget to pass particle types ...
                    'notype' if labels is None else labels[row_index],

                    # ... followed by all the numerical column data ...
                    *[str(entry) for entry in row]

                # ... and a newline, because apparently "writelines" ...
                # ... does not automatically terminate lines with a ...
                # ... linebreak
                ])+'\n'

                # enumerating the rows, because we want to also index ...
                # ... the passed particle labels, if given
                for row_index, row in enumerate(data.T)
            )
