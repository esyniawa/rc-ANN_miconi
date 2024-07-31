import ANNarchy as ann


InputNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0 : population
    """,
    equations="""
    r = baseline + phi * Uniform(-1.0,1.0)
    """
)

OutputNeuron = ann.Neuron(
    parameters="""
    phi = 0.0 : population
    """,
    equations="""
    r = sum(in) + phi * Uniform(-1.0,1.0)
    """
)
MiconiNeuron = ann.Neuron(
    parameters = """
        tau = 30.0 : population # Time constant
        constant = 0.0 # The four first neurons have constant rates
        alpha = 0.05 : population # To compute the sliding mean
        f = 3.0 : population # Frequency of the perturbation
        A = 16. : population # Perturbation amplitude. dt*A/tau should be 0.5...
    """,
    equations="""
        # Perturbation
        perturbation = if Uniform(0.0, 1.0) < f/1000.: 1.0 else: 0.0 
        noise = if perturbation > 0.5: A * Uniform(-1.0, 1.0) else: 0.0

        # ODE for x
        x += dt*(sum(in) + sum(exc) - x + noise)/tau

        # Output r
        rprev = r # store r at previous time step
        r = if constant == 0.0: tanh(x) else: tanh(constant)

        # Sliding mean
        delta_x = x - x_mean
        x_mean = alpha * x_mean + (1 - alpha) * x
    """
)

MiconiLearningRule = ann.Synapse(
    parameters="""
        eta = 0.5 : projection # Learning rate
        learning_phase = 0.0 : projection # Flag to allow learning only at the end of a trial
        reward = 0.0 : projection # Reward received
        mean_reward = 0.0 : projection # Mean Reward received
        max_weight_change = 0.0003 : projection # Clip the weight changes
    """,
    equations="""
        # Trace
        trace += if learning_phase < 0.5:
                    power(pre.rprev * (post.delta_x), 3)
                 else:
                    0.0

        # Weight update only at the end of the trial
        delta_w = if learning_phase > 0.5:
                eta * trace * fabs(mean_reward) * (reward - mean_reward)
             else:
                 0.0 : min=-max_weight_change, max=max_weight_change
        w += delta_w

    """
)
