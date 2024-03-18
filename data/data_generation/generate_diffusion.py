from pde import DiffusionPDE, ScalarField, UnitGrid, CartesianGrid, MemoryStorage, PDE, movie, plot_kymograph
import numpy as np
import gc

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
    results = eq.solve(state, t_range=N_SAMPLES * DT, dt=DT, tracker=[storage.tracker(DT)], )
    for time_step, field in storage.items():
        example.append(np.expand_dims(field.data, axis=0))
    example = np.expand_dims(np.concatenate(example.copy()), axis=0)[:, 1:, ...]
    print(example.mean(), example.std(), example.min(), example.max())
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

def main():
    with np.load('grf_initial_conditions.npz') as file_data:
        intial_conditions = file_data['intial_conditions']
    print(intial_conditions.shape)
    # from generate_grfs import GaussianRF
    # grf = GaussianRF(alpha=1.5, tau=3)
    # intial_conditions = grf.sample(1)[0,...] * 2

    t0 = time.time()

    import multiprocessing
    from tqdm import tqdm
    # solutions = []
    with multiprocessing.Pool(processes=10) as pool:
        solutions = pool.map(run, intial_conditions)
        # for outputs in tqdm(pool.map(run, intial_conditions), total=intial_conditions.shape[0]):
        #     print(outputs.shape)
        #     solutions.append(outputs)
    print(time.time() - t0)
    print(len(solutions))
    solutions = np.concatenate(solutions)
    print(solutions.shape)

    if(PERIODIC):
        name = f'heat_equation_{DIFFUSIVITY}_periodic.npz'
    else:
        name = f'heat_equation_{DIFFUSIVITY}_non_periodic.npz'
    np.savez(
        name,
        solutions=solutions,
        intial_conditions=intial_conditions,
        time_steps=np.array([(t+1) * DT for t in range(N_SAMPLES)]),
        diffusivity=np.array([0.5,]),
    )

    # solutions = []
    # for idx in range(intial_conditions.shape[0]):
    #     eq, state = generate_equation(intial_conditions[idx,...], periodic=True)
    #     ans = solve_pde(eq, state)
    #     solutions.append(ans)
    # solutions = np.concatenate(solutions)
    # print(solutions.shape)
    
if __name__ == '__main__':
    main()