from math import log
import numpy as np
import numpy.linalg as lin
import math

def _normalize_prob(prob, item_set):
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
        else:
            for item in item_set:
                result[item] = 0

    return result 

def _normalize_prob_two_dim(prob, item_set1, item_set2):
    result = {}
    if prob is None:
        for item in item_set1:
            result[item] = _normalize_prob(None, item_set2)
    else:
        for item in item_set1:
            result[item] = _normalize_prob(prob.get(item), item_set2)

    return result

def _count(item, count):
    if item not in count:
        count[item] = 0
    count[item] += 1

def _count_two_dim(item1, item2, count):
    if item1 not in count:
        count[item1] = {}
    _count(item2, count[item1])

def gaussian(x, mean, cov, correction = False):
    x = np.array(x)
    D = x.shape[0]
    
    coeff = 1 / ((2 * math.pi) ** (D / 2) * lin.det(cov) ** 0.5)
    _exp = math.exp(-0.5 * (x - mean).T @ lin.inv(cov) @ (x - mean))
    
    if correction == False:
        return coeff * _exp
    
    elif correction == True:
        n = coeff * _exp
        m = abs(int(math.log10(n)))
        
        return n * 10 ** m

# HMM model with multivariate gaussian emission distribution 

class Model(object):

    # emission_prob: {state1: (mu1, cov1), ...}
    def __init__(self, states, start_prob=None, tran_prob=None, emit_prob=None):
        self._states = set(states)
        self._start_prob = _normalize_prob(start_prob, self._states)
        self._trans_prob = _normalize_prob_two_dim(tran_prob, self._states, self._states)
        # multivariate gaussian distribution 
        self._emit_prob = emit_prob
        
    def _forward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        alpha = [{}]
        for state in self._states:
            element = self._emit_prob[state]
            gauss = gaussian(x=sequence[0], mean=element[0], cov=element[1])
            alpha[0][state] = self._start_prob[state] * gauss

        for index in range(1, sequence_length):
            alpha.append({})
            for state_to in self._states:
                prob = 0
                for state_from in self._states:
                    prob += alpha[index - 1][state_from] * \
                        self._trans_prob[state_from][state_to]
                element = self._emit_prob[state_to]
                gauss = gaussian(x=sequence[index], mean=element[0], cov=element[1])
                alpha[index][state_to] = prob * gauss

        return alpha

    def _backward(self, sequence):
        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        beta = [{}]
        for state in self._states:
            beta[0][state] = 1

        for index in range(sequence_length - 1, 0, -1):
            beta.insert(0, {}) 
            for state_from in self._states:
                prob = 0
                for state_to in self._states:
                    element = self._emit_prob[state_to]
                    gauss = gaussian(x=sequence[index], mean=element[0], cov=element[1])
                    prob += beta[1][state_to] * \
                        self._trans_prob[state_from][state_to] * gauss
                beta[0][state_from] = prob

        return beta

    def evaluate(self, sequence):
        length = len(sequence)
        if length == 0:
            return 0

        prob = 0
        alpha = self._forward(sequence)
        
        for state in alpha[length - 1]:
            prob += alpha[length - 1][state]

        return prob

    def viterbi(self, sequence):

        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        delta = {}
        #print('index:', 0)
        for state in self._states:
            element = self._emit_prob[state]
            gauss = gaussian(x=sequence[0], mean=element[0], cov=element[1])
            delta[state] = self._start_prob[state] * gauss
            #print('gauss:', gauss, 'start_prob:', self._start_prob[state])
        #print('delta:', delta)
        
        pre = []
        for index in range(1, sequence_length):
            #print('index:', index)
            delta_bar = {}
            pre_state = {}
            for state_to in self._states:
                max_prob = 0
                max_state = None
                for state_from in self._states:
                    prob = delta[state_from] * self._trans_prob[state_from][state_to]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from
                element = self._emit_prob[state_to]
                gauss = gaussian(x=sequence[index], mean=element[0], cov=element[1])
                #print('gauss:', gauss)
                delta_bar[state_to] = max_prob * gauss
                pre_state[state_to] = max_state
            delta = delta_bar
            #print('delta:', delta)
            pre.append(pre_state)
        max_state = None
        max_prob = 0
        for state in self._states:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state
        if max_state is None:
            raise Exception('delta 값이 너무 작아서 0으로 간주되었습니다.')

        # 상태 역순 추적
        result = [max_state]
        for index in range(sequence_length - 1, 0, -1):
            max_state = pre[index - 1][max_state]
            result.insert(0, max_state)

        return result

    def baumwelch(self, sequence, smoothing=0):
        length = len(sequence)
        alpha = self._forward(sequence)
        beta = self._backward(sequence)
        gamma = []
        for index in range(length):
            prob_sum = 0
            gamma.append({})
            for state in self._states:
                prob = alpha[index][state] * beta[index][state]
                gamma[index][state] = prob
                prob_sum += prob

            if prob_sum == 0:
                continue

            for state in self._states:
                gamma[index][state] /= prob_sum
        xi = []
        for index in range(length - 1):
            prob_sum = 0
            xi.append({})
            for state_from in self._states:
                xi[index][state_from] = {}
                for state_to in self._states:
                    element = self._emit_prob[state_to]
                    gauss = gaussian(x=sequence[index+1], mean=element[0], cov=element[1])
                    prob = alpha[index][state_from] * beta[index + 1][state_to] * \
                        self._trans_prob[state_from][state_to] * gauss
                    xi[index][state_from][state_to] = prob
                    prob_sum += prob

            if prob_sum == 0:
                continue

            for state_from in self._states:
                for state_to in self._states:
                    xi[index][state_from][state_to] /= prob_sum
        states_number = len(self._states)
        for state in self._states:
            # update start probability
            self._start_prob[state] = \
                (smoothing + gamma[0][state]) / (1 + states_number * smoothing)
            
            # update transition probability
            gamma_sum = 0
            for index in range(length - 1):
                gamma_sum += gamma[index][state]

            if gamma_sum > 0:
                denominator = gamma_sum + states_number * smoothing
                for state_to in self._states:
                    xi_sum = 0
                    for index in range(length - 1):
                        xi_sum += xi[index][state][state_to]
                    self._trans_prob[state][state_to] = (smoothing + xi_sum) / denominator
            else:
                for state_to in self._states:
                    self._trans_prob[state][state_to] = 0
            
            # update emission probability
            gamma_sum += gamma[length - 1][state]
            emit_gamma_sum = {}
            mean_upper = 0
            cov_upper = 0
            for index in range(length):
                mean_upper += gamma[index][state] * sequence[index]
            new_mean = mean_upper / gamma_sum
            for index in range(length):
                dev = sequence[index] - new_mean
                cov_upper += gamma[index][state] * np.array([dev]).T @ np.array([dev])
            new_cov = cov_upper / gamma_sum
            
            if gamma_sum > 0:
                self._emit_prob[state] = (new_mean, new_cov)
            else:
                raise Exception('gamma_sum의 값이 0입니다.')