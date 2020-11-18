from EOG_HMM import Model
import numpy as np
from preprocess import preprocess

a1 = preprocess('/Users/aiden0206/Desktop/Domain/EOG/EOG 데이터셋/ud/mudududm1.txt', 2)
a1.normalize([5, 70])
ch1 = a1.linear_baseline(a1.norm['channel0'])
ch2 = a1.linear_baseline(a1.norm['channel1'])
seq = np.array([ch1, ch2]).T

states = ('up', 'down', 'middle')
start_prob = {'up': 0.333, 'down': 0.333, 'middle': 0.333}
transition_prob = {'up': {'up': 0.5, 'middle': 0.5}, 'down': {'down': 0.5, 'middle': 0.5}}
mu1 = [0, 0]
mu2 = [-0.3, -0.4]
mu3 = [0.6, 0.4]
cov1 = [[0.5, 0.0], [0.0, 0.5]]
cov2 = [[0.5, 0.0], [0.0, 0.5]]
cov3 = [[0.5, 0.0], [0.0, 0.5]]
emitting_prob = {'middle': (mu1, cov1), 'up': (mu2, cov2), 'down': (mu3, cov3)}

ud_test = Model(states, start_prob, transition_prob, emitting_prob)

sep_data = []
for window in range(1000 // 2):
    sep_data.append(ud_test.viterbi(sequence = seq[window*2 : (window + 1)*2]))

# plot
figs = plt.figure(figsize=(20, 15))
ax1 = figs.add_subplot(311)
ax2 = figs.add_subplot(312)
ax3 = figs.add_subplot(313)
for i in range(1000 // 10):
    location = 0.04 * i
    text = sep_data[i*(5)][1]
    if text == 'middle':
        ax1.annotate('m', xy=(location, 1), xytext=(location, 1.5), arrowprops = dict(facecolor='black'))    
    elif text == 'up':
        ax1.annotate('u', xy=(location, 1), xytext=(location, 1.5), arrowprops = dict(facecolor='red'))
    elif text == 'down':
        ax1.annotate('d', xy=(location, 1), xytext=(location, 1.5), arrowprops = dict(facecolor='blue'))

x = np.linspace(0, 4, 1000)
ax1.plot(x, x)
ax2.plot(x, ch1)
ax3.plot(x, ch2)
plt.show()