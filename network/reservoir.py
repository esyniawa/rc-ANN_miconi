import numpy as np
import ANNarchy as ann
ann.setup(num_threads=4)

import os
import matplotlib.pyplot as plt

from typing import Optional
from .definitions import InputNeuron, OutputNeuron, MiconiNeuron, MiconiLearningRule


class MiconiReservoir:
    def __init__(self,
                 N_res: int = 500,
                 N_in: int = 2,
                 N_out: int = 2,
                 rho: float = 1.5,
                 sigma_rec: float = 0.1):

        # network parameters
        self.N_res = N_res
        self.N_in = N_in
        self.N_out = N_out

        # monitors for recording
        self.monitors = {}
        self.sample_rate = 1.0

        self.network = self.build_network(g=rho, sparseness=sigma_rec)

    def build_network(self,
                      g: float,
                      sparseness: float):

        # Input population
        input_pop = ann.Population(self.N_in, neuron=InputNeuron, name='Input')

        # Recurrent population
        res_pop = ann.Population(self.N_res, neuron=MiconiNeuron, name='Reservoir')

        # Biases
        res_pop[0].constant = 1.0
        res_pop[1].constant = 1.0
        res_pop[2].constant = -1.0

        out_pop = ann.Population(self.N_out, neuron=OutputNeuron, name='Output')

        # Input weights
        Wi = ann.Projection(input_pop, res_pop, target='in')
        Wi.connect_all_to_all(weights=ann.Uniform(-1.0, 1.0))

        # Recurrent weights
        Wrec = ann.Projection(res_pop, res_pop, target='exc', synapse=MiconiLearningRule, name='MiconiSynapse')
        if sparseness == 1.0:
            Wrec.connect_all_to_all(weights=ann.Normal(0., g / np.sqrt(self.N_res)))
        else:
            Wrec.connect_fixed_probability(probability=sparseness,
                                           weights=ann.Normal(0., g / np.sqrt(sparseness * self.N_res)))

        # Pick the output neurons
        for i in range(self.N_out):
            id_out = self.N_res - i - 1
            proj = ann.Projection(pre=res_pop[id_out], post=out_pop[i], target='in')
            proj.connect_one_to_one(1.0)

        monitor_out = ann.Monitor(out_pop, variables='r', period=self.sample_rate, start=False)
        self.monitors[out_pop.name] = monitor_out

        network = ann.Network(everything=True)

        return network

    def compile_network(self, folder: str):
        self.network.compile(directory='annarchy/' + folder)

    def get_reservoir(self):
        return self.network.get_population('Reservoir')

    def get_synapse(self):
        return self.network.get_projection('MiconiSynapse')

    def get_input(self):
        return self.network.get_population('Input')

    def get_output(self):
        return self.network.get_population('Output')

    def extract_recurrent_weights(self):
        proj = self.get_synapse()
        weights = np.zeros((self.N_res, self.N_res))
        for dendrite in proj:
            for pre_id, weight in zip(dendrite.pre_rank, dendrite.w):
                weights[pre_id, dendrite.post_rank] = weight
        return weights

    def init_monitor(self,
                     pop_name: str,
                     var_names: str | list[str] = 'r',
                     start: bool = False,
                     sample_rate: float = 2.0):

        m = ann.Monitor(self.network.get_population(pop_name), variables=var_names, start=start, period=sample_rate)
        self.network.add([m])
        self.monitors[pop_name] = m

    def start_monitors(self):
        for m in self.monitors.values():
            self.network.get(m).start()

    def pause_monitors(self):
        for m in self.monitors.values():
            self.network.get(m).pause()

    def resume_monitors(self):
        for m in self.monitors.values():
            self.network.get(m).resume()

    def get_monitor(self,
                    pop_name: str,
                    var_name: str = 'r',
                    keep: bool = False,
                    reshape: bool = False):
        return self.network.get(self.monitors[pop_name]).get(keep=keep, reshape=reshape)[var_name]

    def run_target(self,
                   data_target: np.ndarray,
                   folder: str,
                   alpha: float = 0.8,
                   d_stim: float = 200,
                   d_response: float = 200,
                   reinitialize: bool = True,
                   perturbation: bool = True,
                   plot: bool = False):

        # get populations
        inp = self.get_input()
        pop = self.get_reservoir()
        out = self.get_output()

        pop_synapse = self.get_synapse()

        # compile network
        self.compile_network(folder=folder)

        R_mean = 0.0
        error_history = []
        eigvals = []

        # Switch off perturbations if needed
        if not perturbation:
            old_A = pop.A
            pop.A = 0.0

        # Run
        for t in range(data_target.shape[0]):

            if reinitialize:
                pop.x = ann.Uniform(-0.1, 0.1).get_values(self.N_res)
                pop.r = np.tanh(pop.x)
                pop[0].r = np.tanh(1.0)
                pop[1].r = np.tanh(1.0)
                pop[2].r = np.tanh(-1.0)

            # First input
            inp.baseline = data_target[t]
            ann.simulate(d_stim, net_id=self.network.id)

            # Response
            self.start_monitors()
            inp.baseline = 0.0
            ann.simulate(d_response, net_id=self.network.id)

            # Read the output
            self.pause_monitors()
            output = self.get_monitor(pop_name=out.name, var_name='r', keep=False)

            # Compute the reward as the opposite of the absolute error
            error = np.linalg.norm(data_target[t] - np.mean(output, axis=0))

            # The first 25 trial do not learn, to let R_mean get realistic values
            if t > 25:
                # Apply the learning rule
                pop_synapse.learning_phase = 1.0
                pop_synapse.reward = error
                pop_synapse.mean_reward = R_mean

                # Learn for one step
                ann.step(net_id=self.network.id)

                # Reset the traces
                pop_synapse.learning_phase = 0.0
                pop_synapse.trace = 0.0

                # Update mean reward
                R_mean = alpha * R_mean + (1. - alpha) * error

                # Store weight + error history
                eigvals.append(self.calculate_eigenvals(self.extract_recurrent_weights()))
                error_history.append(error)

        if plot:
            if not os.path.exists('figures/' + folder):
                os.makedirs('figures/' + folder)


            fig, axs = plt.subplots(nrows=2)
            axs[0].plot(np.array(error_history), label='Error')
            axs[1].plot(np.array(eigvals), label='Eigenvalues')
            plt.legend()
            plt.savefig('figures/' + folder + 'error.png')
            plt.savefig('figures/' + folder + 'error.png')

        # Switch back on perturbations if needed
        if not perturbation:
            pop.A = old_A


    @staticmethod
    def make_dynamic_target(dim_out: int,
                            n_trials: int,
                            low_bound: int = 50,
                            high_bound: int = 150,
                            seed: Optional[int] = None):
        """
        Generates a dynamic target signal for the reservoir computing network.

        :param dim_out: The dimensionality of the output signal.
        :param n_trials: The number of trials for which the signal is generated.
        :param seed: The seed for the random number generator. Default is None.

        :return: A tuple containing the generated dynamic target signal (numpy array) and the period time (float).
        """

        # random period time
        T = np.random.RandomState(seed).uniform(low_bound, high_bound)
        x = np.arange(0, n_trials * T)

        y = np.zeros((len(x), dim_out))

        for out in range(dim_out):
            a1 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a2 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a3 = np.random.RandomState(seed).normal(loc=0, scale=0.5)

            y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

        return y, T

    @staticmethod
    def calculate_eigenvals(mat: np.ndarray):
        return np.linalg.eigvalsh(mat)
