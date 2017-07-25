###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Aug-09-2016
# Revised:
###############################################################################
# Standardize the data file used for partial wave analysis.
###############################################################################

from numpy import int64, float64
from sympy import Float, Integer


class DataFile:
    """Permit easy access and standardized exportability to a data file.

    Attributes
    ----------
    header: str
            Any comments related to the file.
    sections: str list
              The description of each data column.
    data: int/float array
          Holds data of file in 2D array.
    """

    ###########################################################################
    # Magic Methods
    ###########################################################################

    def __init__(self):
        self.data = []
        self.sections = []
        self.header = ""
        self.metadata = []

    def __getitem__(self, keys):
        """Return columns and rows specified by keys.

        Parameters
        ----------
        one integer: return corresponding column.
        two integers: first/second correspond to rows/columns, respectively.
        """
        if isinstance(keys, tuple):
            row_key, col_key = keys
            rows = self.data[row_key]
        else:
            col_key = keys
            rows = self.data
        cols = self.column(rows, col_key)
        return cols

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        """Return number of columns of data"""
        return len(self.data[0])

    ###########################################################################
    # Methods
    ###########################################################################

    @staticmethod
    def column(array, key):
        if isinstance(array[0], list):
            col = [row[key] for row in array]
        else:
            col = array[key]
        return col

    @staticmethod
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    def add_column(self, values):
        """Append column to self.data from 1D values array.
        """
        if self.data == []:
            self.data = [[item] for item in values]
        else:
            assert len(values) == len(self.data), "Wrong number of data."
            for i, item in enumerate(values):
                self.data[i].append(item)

    def read(self, file, delim=None):
        self.data = []
        self.metadata = []
        with open(file, 'r') as f:
            for line in f:
                if line[0] == "#":
                    # Append everything after "# "
                    self.metadata.append(line[2:])
                else:
                    entries = line.split(delim)
                    self.data.append([self.num(entry) for entry in entries])
        return self

    def write(self, *data_tuples, header="", metadata=[]):
        self.data = []
        self.sections = []
        self.header = header
        self.metadata = metadata
        for item in data_tuples:
            if isinstance(item, tuple):
                self.sections.append(item[0])
                values = item[1]
            else:
                self.sections.append("")
                values = item
            self.add_column(values)
        return self

    def export_to_file(self, file_name, is_scientific=False):
        with open(file_name, "w+") as f:
            # Set up section and data formatting.
            sec_string = "# "
            data_string = "  "
            for i, data in enumerate(self.data[0]):
                if isinstance(data, int) or isinstance(data, int64) or isinstance(data, Integer):
                    # Int columns consume of total of 10 characters.
                    sec_string += "{sec[" + str(i) + "]:^10s}"
                    data_string += "{row[" + str(i) + "]:5d}     "
                elif isinstance(data, float) or isinstance(data, float64) or isinstance(data, Float):
                    # Float columns consume of total of 26 characters.
                    sec_string += "{sec[" + str(i) + "]:^25s}"
                    if is_scientific:
                        data_string += "   {row[" + str(i) + "]:.15e}   "
                    else:
                        data_string += "   {row[" + str(i) + "]:19.15f}   "
                else:
                    print("Data type error!", data)
                    return "Data type error!"

            # Write to file
            f.write("# " + self.header + "\n")
            for item in self.metadata:
                f.write("# " + item + "\n")

            try:
                f.write(sec_string.format(sec=self.sections).rstrip())
            except IndexError:
                pass

            for row in self.data:
                f.write("\n")
                f.write(data_string.format(row=row).rstrip())

    ###########################################################################
    # End of DataFile class.
    ###########################################################################
