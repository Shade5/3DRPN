import numpy as np
import binvox_rw


def read_bv(fn):
    with open(fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    data = np.float32(model.data)
    return data
