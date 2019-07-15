#coding:utf-8
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms

# Pre-emphasis filter
def preEmphasis(signal, p):
  return scipy.signal.lfilter([1.0, -p], 1, signal)

# Mel Filtert Bnak
def melFilterBank(fs, nfft, numChannels):
    # Nyquist frequency(Hz)
    fmax = fs / 2
    # Nyquist frequency(mel)
    melmax = hz2mel(fmax)
    nmax = nfft / 2
    df = fs / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
    filterbank = np.zeros((numChannels, int(nmax)))
    for c in np.arange(0, numChannels):
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, int(i)] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)
    return filterbank, fcenters

def hz2mel(f):
  # Hz -> mel
  return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
  # mel -> hz
  return 700.0 * (np.exp(m / 1127.01048) - 1.0)

# filename: file name
# nfft: number of FFT ( 1024, 2048, 4096 )
# nceps: number of demention( many case: 12 )
def mfcc(filename,nfft,nceps):

	# Preprocessing
	wf = wave.open(filename,'r')
	fs = float(wf.getframerate())
	wav = wf.readframes(wf.getnframes())
	wav = np.frombuffer(wav,dtype='int16')/32768.0 # normalization (-1,1) , int16:[-32,768~32,767]
	wf.close()
	t = np.arange(0.0, len(wav)/fs, 1/fs)
	center = len(wav)/2
	cuttime = 0.08
	wavdata = wav[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]

	# Pre-emphasis filter
	p = 0.97 # Pre-emphasis
	signal = preEmphasis(wavdata,p)

	# Hamming window
	hammingWindow = np.hamming(len(signal))
	signal = signal * hammingWindow

	# Amplitude spectrum
	spec = np.abs(np.fft.fft(signal, nfft))[:nfft//2]
	fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:nfft//2]

	# Mel filter bank
	numChannels = 20 # Number of channels of Mel filter bank
	df = fs / nfft # Frequency resolution
	filterbank, fcenters = melFilterBank(fs, nfft, numChannels)

	mspec = np.log10(np.dot(spec, filterbank.T))
	ceps = scipy.fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)
	return ceps[:nceps].tolist()
 
