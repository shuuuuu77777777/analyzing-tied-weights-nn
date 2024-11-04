import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0,6 ,1000)

ma = 10
y = 3



lamba = 1

delta = 1/np.e

jyurai = []
proposed = []
proposed_delta = [] #
xlogx = []


for i in range(len(r)):
    jyurai.append((y )* np.log((y) / (ma  + lamba*r[i])) + ma - y + r[i])
    proposed.append((y - r[i]) * np.log((y - r[i]) / ma) + ma - y + r[i])
    proposed_delta.append((y  + delta- r[i]) * np.log((y + delta - lamba*r[i]) / (ma+delta)) + ma - y + r[i])
    xlogx.append(r[i]*np.log(r[i]))


plt.plot(r, jyurai, label="jyurai", color='blue')
plt.plot(r, proposed, label="proposed", color='green')
plt.plot(r, proposed_delta, label="proposed_delta", color='red')
plt.plot(r, r, label="r", color='black')
# plt.plot(r,xlogx,label = 'xlogx', color = 'red')
# plt.plot(r, proposed + r, label="propsosed loss", color='green')
# plt.plot(r, jyurai + r, label="jyurai loss", color='blue')
# plt.plot(r, proposed_delta + r, label="proposed_deltaloss", color='red')

plt.xlabel('r')
plt.ylabel('Values')
plt.title('Comparison of jyurai and proposed')
plt.legend()

plt.show()

