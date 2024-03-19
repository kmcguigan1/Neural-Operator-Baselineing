from pde import DiffusionPDE, ScalarField, UnitGrid, CartesianGrid, MemoryStorage, PDE, movie, plot_kymograph
import numpy as np
import gc

import h5py

import time

DT = 0.5
N_SAMPLES = 50
PERIODIC = True
DIFFUSIVITY = 0.5

def generate_equation(intial_conditions:np.ndarray, periodic:bool=PERIODIC, diffusivity:float=DIFFUSIVITY):
    grid = UnitGrid(intial_conditions.shape, periodic=[periodic, periodic])
    state = ScalarField(grid, data=intial_conditions)
    if(periodic):
        boundary_conditions = ['periodic', 'periodic']
    else:
        boundary_conditions = [{"derivative": 0.0}, {"derivative": 0.0}]
    eq = DiffusionPDE(diffusivity=diffusivity, bc=boundary_conditions)
    return eq, state

def solve_pde(eq, state):
    example = []
    storage = MemoryStorage()
    _ = eq.solve(state, t_range=N_SAMPLES * DT, dt=DT, tracker=[storage.tracker(DT)], )
    for time_step, field in storage.items():
        example.append(np.expand_dims(field.data, axis=0))
    example = np.expand_dims(np.concatenate(example.copy()), axis=0)[:, 1:, ...]
    # print(example.mean(), example.std(), example.min(), example.max())
    del storage
    # movie(storage, 'example.mov', plot_args={'cmap':'rainbow'}, movie_args={'framerate':10})
    return example

def run(args):
    intial_conditions = args
    eq, state = generate_equation(intial_conditions)
    ans = solve_pde(eq, state)
    del eq
    del state
    gc.collect()
    return ans

import multiprocessing
from tqdm import tqdm

# EXAMPLE_COUNT_PER_RUN = 1000
# CHUNK_SIZE = 15

def main(split='train'):
    with np.load('grf_initial_conditions.npz') as file_data:
        intial_conditions = file_data[f'{split}_conditions']
    print(intial_conditions.shape)

    if(PERIODIC):
        name = 'heat_equation_periodic.h5'
    else:
        name = 'heat_equation_non_periodic.h5'

    t0 = time.time()
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(run, intial_conditions)
    print(time.time() - t0)
    
    results = np.concatenate(results).astype(np.float32)
    print(results.shape)
    with h5py.File(name, "a") as f:
        f.create_dataset(
            f'{split}_u', 
            data=results, 
        )
        f.create_dataset(
            f'{split}_a', 
            data=intial_conditions, 
        )
        f.create_dataset(
            f'{split}_t', 
            data=np.array([(t + 1) * DT for t in range(N_SAMPLES)]), 
        )

# def main(run_idx=0):
#     with np.load('grf_initial_conditions.npz') as file_data:
#         intial_conditions = file_data['intial_conditions']
#     example_lower_bound = EXAMPLE_COUNT_PER_RUN * run_idx
#     example_upper_bound = min(intial_conditions.shape[0], example_lower_bound + EXAMPLE_COUNT_PER_RUN)
#     intial_conditions = intial_conditions[example_lower_bound:example_upper_bound, ...]
#     print(intial_conditions.shape)

#     if(PERIODIC):
#         name = 'heat_equation_periodic.h5'
#     else:
#         name = 'heat_equation_non_periodic.h5'

#     with h5py.File(name, "a") as f:
#         if(run_idx == 0):
#             dset = f.create_dataset(
#                 'solution', 
#                 shape=(CHUNK_SIZE, N_SAMPLES, *intial_conditions.shape[1:]), 
#                 maxshape=(None, N_SAMPLES, *intial_conditions.shape[1:]),
#                 dtype='float32',
#                 chunks=(CHUNK_SIZE, N_SAMPLES, *intial_conditions.shape[1:])
#             )
#             row_count = 0
#         else:
#             dset = f['solution']
#             row_count = dset.shape[0]


#         t0 = time.time()
#         chunk_idx = 0
#         chunk = np.empty((CHUNK_SIZE, N_SAMPLES, *intial_conditions.shape[1:]))
#         with multiprocessing.Pool(processes=8) as pool:
#             for output_idx, outputs in enumerate(pool.imap(run, intial_conditions, chunksize=5), start=1):
#                 chunk[chunk_idx, ...] = outputs
#                 chunk_idx += 1
#                 if(chunk_idx == CHUNK_SIZE or output_idx==intial_conditions.shape[0]):
#                     assert np.all(np.isnan(chunk)) == False
#                     dset.resize(row_count + chunk.shape[0], axis=0)
#                     dset[row_count:] = chunk
#                     row_count += chunk.shape[0]
#                     remainder = min(CHUNK_SIZE, intial_conditions.shape[0] - output_idx)
#                     chunk = np.empty((remainder, N_SAMPLES, *intial_conditions.shape[1:]))
#                     chunk_idx = 0
#     print(time.time() - t0)
#     gc.collect()
    
if __name__ == '__main__':
    main()