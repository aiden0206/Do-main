from EOG_HMM import Model
from math import log

def train(states, initial_start, initial_tran, initial_emit, sequences, delta=0.0001, smoothing=0):

    model = Model(states=states, start_prob=initial_start, tran_prob=initial_tran, emit_prob=initial_emit)
    old_likelihood = 0
    old_likelihood += log(model.evaluate(sequences))

    while True:
        new_likelihood = 0
        model.baumwelch(sequences)
        new_likelihood += log(model.evaluate(sequences))

        if abs(new_likelihood - old_likelihood) < delta:
            break

        old_likelihood = new_likelihood

    return model