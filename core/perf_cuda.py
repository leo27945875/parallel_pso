import cuPSO

import timeit
import random
import numpy as np

from .pycupso import PSO_CUDA
from .funcs import *
from .animation import *


def main() -> None:

    seed    = 0
    dim     = 1024
    n       = 1024
    iters   = 1000
    x_min   = -20
    x_max   = 20.
    v_max   = 1.
    n_test  = 30
    device  = cuPSO.Device.GPU

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pso = PSO_CUDA(
        dim    = dim,
        n      = n,
        iters  = iters,
        x_min  = x_min,
        x_max  = x_max,
        v_max  = v_max,
        seed   = seed,
        device = device
    )

    t = timeit.timeit(lambda: pso.run(), number=n_test) / n_test
    print(f"Total time = {t}(s)")

    
if __name__ == "__main__":

    main()