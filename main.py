from sys import argv
n_sims = int(argv[1])

from network.reservoir import MiconiReservoir

def main(sims: int, n_out=2):
    for sim in range(sims):
        reservoir = MiconiReservoir(N_in=n_out, N_out=n_out)
        targets, _ = MiconiReservoir.make_dynamic_target(dim_out=n_out, n_trials=1_000)
        reservoir.run_target(folder=f'run_{sim}/', data_target=targets, plot=True)

if __name__ == '__main__':
    main(sims=n_sims)