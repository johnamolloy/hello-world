# -*- coding: utf-8 -*-
"""
30/01/2018

cd \d_Development\Python\Scripts\kata\Kata_04_WavFileFFT

@author: johnm
"""


#%%

import numpy as np
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
from scipy.fftpack import fft


#%%

# - Read the wav file into numpy array 'w'.
fs, w = wv.read('LovetFaller.wav')

# - Convert to a float.
w = w.astype('float64')

# - Show the number of channels, in some way.
if len(w.shape) == 1:
    print("Number of Channels: 1")
else:
    print("Number of Channels: {0}".format(w.shape[1]))

# - If more than one channel, convert to mono using a mean.
if len(w.shape) == 2:
    w = w.mean(axis=1)

# - Normalise on [-1.0, 1.0].
w = w - np.mean(w)
w /= max(abs(w))

# - Visualise the data, using an appropriate step.
plt.plot(w[::10])

# - Choose a start point and time period to analyse
# (JoMo: Consider the power of two thing.)
t = 10.0
dt = 0.25

N = int(dt * fs)
N = N - N % 2
NN = int(N / 2)

print("N: {0}".format(N))

# - Display that period


#%%
# Check for drop-outs.

nstep = 100

plt.plot(w[::nstep])

n = 4
fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8,2*n))
fig.subplots_adjust(hspace=0)

q = int(w.shape[0] / n)

for i in range(n):
    axs[i].plot(w[i*q:(i+1)*q:nstep])
    axs[i].set_yticks([])

plt.show()

del i, n, nstep, q
del axs, fig


#%%
# Fourier trasform of a specific slice of time.

t = 16.0
dt = 0.25

N = int(dt * f)
N = N - N % 2
NN = int(N / 2)

ws = w[int(t*f):int(t*f)+N]
ws -= np.mean(ws)
ws /= max(abs(ws))

yf = fft(ws)
yf = np.abs(yf)
yf = yf[:NN]
yf /= NN        # JoMo: Check this

xf = np.linspace(0.0, f/2, NN)

maxf = 2000

n = int(maxf * N / f)

xf = xf[:n]
yf = yf[:n]

#plt.plot(xf[:n], yf[:n])
plt.plot(xf, yf)
#plt.ylim(0, 0.1)


#%%
#
# Recreate the sounds, in chunks
#

dt     = 0.25
tstart = 15.0
tend   = 30.0

#dt     = 0.01
#tstart = 53.0
#tend   = 63.0

thresh = 0.1

maxf = 2000

N = int(dt * f)
N = N - N % 2
NN = int(N / 2)

n = int(maxf * N / f)

xf = np.linspace(0.0, f/2, NN)
xf = xf[:n]

tseg = np.linspace(0, dt, N)

env = np.sin(np.linspace(0, np.pi, N))

wr = np.array([])

# Smoothing envelope
# Type 1
env = np.sin(np.linspace(0, np.pi, N)) ** 1
# Type 2
#frac = 10
#env = np.zeros(N) + 1.0
#env[:int(N/frac)] = np.linspace(0, 1, int(N/frac))
#env[-int(N/frac):] = np.linspace(1, 0, int(N/frac))
#env = env**0.1

#fdict = {}

for t in np.arange(tstart, tend, dt):

    print(t)

    # Use the normalised data.
    ws = wn[int(t*f):int(t*f)+N]
    
    yf = fft(ws)
    yf = np.abs(yf)
    yf = yf[:NN]
    yf /= NN
    
    yf = yf[:n]
    
    threshold = thresh * max(yf)   # Bear in mind that this will vary between segments

    q = zip(xf[yf >= threshold], yf[yf >= threshold])
    #fdict[t] = (threshold, len(yf[yf >= threshold]), q)

    wseg = np.zeros(N)
    for freq, weight in q:
        wseg += weight * np.sin(2 * np.pi * freq * tseg)
    wseg *= env

    wr = np.append(wr, wseg, axis=0)

wr = wr / abs(wr).max()
wr = (32000 * wr).astype('int16')

# Tidy this...
wx = wn[int(tstart*f):int(tstart*f)+wr.shape[0]]
wx = wx / abs(wx).max()
wx = (32000 * wx).astype('int16')

wmix = np.vstack((wr, wx)).T

wv.write('zz_recreated.wav', f, wr)
wv.write('zz_recreated_cmp.wav', f, wmix)


#%%
#
#
#

dt = 0.25

N = int(dt * f)
N = N - N % 2
NN = int(N / 2)

xf = np.linspace(0.0, f/2, NN)

maxf = 2000
n = int(maxf * N / f)

nchart = 21

fig, axs = plt.subplots(nchart, 1, sharex=True)
fig.subplots_adjust(hspace=0.35)

fig.set_figheight(2 * nchart)
fig.set_figwidth(10)

i = 0
for t in np.linspace(15, 20, nchart):

    ws = w[int(t*f):int(t*f)+N]
    ws -= np.mean(ws)
    ws /= max(abs(ws))

    yf = fft(ws)
    yf = np.abs(yf)
    yf = yf[:NN]
    yf /= NN        # JoMo: Check this

    axs[i].plot(xf[:n], yf[:n])
    #axs[i].bar(range(q.shape[1]), q[i], 0.99)
    #axs[i].set_yticks([0, np.ceil(np.max(yf[:n]))])
    #axs[i].set_yticks([])
    #axs[i].set_ylim(0, 1)
    axs[i].set_title("t={0:.2f}".format(t))
    
    i += 1

plt.show()


#%%
#
# Problems
# 1) This fails if more than one item has the maximum value.
#    See the example in the next cell.
# 2) The sum of the mid-point an neighbouring points is 
#    incorrect if di=1 and there are peaks at i and i+2, 
#    because the values at point i+1 is shared twice, between
#    the two points.
#

yq = np.zeros(n)
yr = np.zeros(n)

m = 1.5
di = 1

for i in range(di, n-di):
    if yf[i] == max(yf[i-di:i+di+1]):
        #yq[i] = yf[i]
        yq[i] = sum(yf[i-1:i+2])

for i in range(di, n-di):
    if yf[i] == max(yf[i-di:i+di+1]):
        if yf[i] > m * np.mean(yf[i-di:i+di+1]):
            #yr[i] = yf[i]
            yr[i] = sum(yf[i-1:i+2])

sq = 0
nq = 110

plt.bar(range(nq), yf[sq:sq+nq])
#plt.xticks(xf[sq:sq+nq])
plt.show()
plt.bar(range(nq), yq[sq:sq+nq])
plt.show()
plt.bar(range(nq), yr[sq:sq+nq])
plt.show()


#%%

st = 2**(1/12) - 1

print("  Amp\t Semi\t    f\t   2f\t   3f\t   4f")
print("-----\t-----\t-----\t-----\t-----\t-----")

# Why round the numbers if they are then displayed to a 
# certain number of decimal places?
# Am I planning to look for integer multiples?

for i in range(len(yr)):
    if yr[i] > 0:
        print("{0:5.0f}\t{1:5.0f}\t{2:5.0f}\t{3:5.0f}\t{4:5.0f}\t{5:5.0f}".format(
            round(1000*yr[i], 0), 
            round(st * xf[i], 0),
            round(1 * xf[i], 0),
            round(2 * xf[i], 0),
            round(3 * xf[i], 0),
            round(4 * xf[i], 0)
        ))

del st


#%%

# Keep this example - it shows it failing.
yf = [0,0,1,1,1,5,5,5,5,5,5,1,1,1,0,0]

n = len(yf)
yq = np.zeros(n)

m = 1.5
di = 2

for i in range(di, n-di):
    if yf[i] == max(yf[i-di:i+di+1]):
        if yf[i] > m * np.mean(yf[i-di:i+di+1]):
            yq[i] = max(yf[i-di:i+di+1])

plt.bar(range(n), yf, 0.5)
plt.show()
plt.bar(range(n), yq, 0.5)
plt.show()


#%%

t = 19.5
dt = 1.2

N = int(dt * f)

ws = w[int(t*f):int(t*f)+N]
#ws -= np.mean(ws)
#ws /= max(abs(ws))

plt.plot(ws)


#%%

q1 = np.array([])
q2 = np.array([1,2,3])

q1 = np.append(q1, q2, axis=0)
q1 = np.append(q1, q2, axis=0)
q1 = np.append(q1, q2, axis=0)
                

#%%

frac = 10
env = np.zeros(N) + 1.0
env[:int(N/frac)] = np.linspace(0, 1, int(N/frac))
env[-int(N/frac):] = np.linspace(1, 0, int(N/frac))
env = env**0.1

plt.plot(env)


#%%

#
# End Of File
#