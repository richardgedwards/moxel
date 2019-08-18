import numpy as np
import sys
import tiledb


# Name of the array to create.

def create_array(array_name):
    # Check if the array already exists.
    # if tiledb.object_type(uri=array_name) == "array":
    #     print("Array already exists.")
    #     sys.exit(0)

    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4].
    dom = tiledb.Domain(tiledb.Dim(name="rows", domain=(1, 4), tile=4, dtype=np.int32),
                        tiledb.Dim(name="cols", domain=(1, 4), tile=4, dtype=np.int32))

    # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                attrs=[tiledb.Attr(name="a", dtype=np.int32)])

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array(array_name, data):
    with tiledb.DenseArray(array_name, mode='w') as A:
        A[:] = data

if __name__=="__main__":
    array_name = "quickstart_dense"

    create_array(array_name)

    # data = np.array(([1, 2, 3, 4],
    #              [5, 6, 7, 8],
    #              [9, 10, 11, 12],
    #              [13, 14, 15, 16]))                       

    # write_array(array_name, data)

    # with tiledb.DenseArray(array_name, mode='r') as A:
    #     # Slice only rows 1, 2 and cols 2, 3, 4.
    #     data = A[:]
    # print(data["a"])        