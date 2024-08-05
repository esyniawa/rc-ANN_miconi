import numpy as np
from sys import argv
n_sims = int(argv[1])

from network.reservoir import MiconiReservoir

def main(sims: int, n_out=1):
    for sim in range(sims):
        reservoir = MiconiReservoir(N_in=n_out, N_out=n_out)
        # targets, _ = MiconiReservoir.make_dynamic_target(dim_out=n_out, n_trials=1_000)
        targets = 0.1 * np.ones((5_000, n_out))
        my_input = 0.6 * np.ones((5_000, n_out))
        reservoir.run_target(folder=f'run_{sim}/',
                             data_target=targets,
                             data_input=my_input,
                             plot=True)

if __name__ == '__main__':
    main(sims=n_sims)