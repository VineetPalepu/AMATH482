import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
from numpy.fft import fft, fftshift, ifft, ifftshift

## Helper Functions

# Converts a given frequency into the corresponding note name
def freq2note(freq):
    if freq < 25 or freq > 4190:
        return ""
    key_num = int(round(12 * np.log2(freq / 440) + 49))
    octave = int(np.floor((key_num + 8) / 12))
    note_index = key_num % 12 - 1
    if note_index == -1:
        note_index = 11
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    return notes[note_index] + str(octave)

# Converts a given note name into the frequency
def note2freq(note):
    if len(note) == 3:
        note_name = note[0:2]
        octave = int(note[2])
    elif len(note) == 2:
        note_name = note[0]
        octave = int(note[1])
    else:
        return 0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    note_index = notes.index(note_name)
    key_num = note_index + 4 + (12 * (octave - 1))
    return np.power(2, (key_num - 49) / 12) * 440

# Convert a frequency into the corresponding index in order to set limits
# on the y-axis of the spectrogram
def getFreqIndex(ks, freq):
    index = ks.searchsorted(freq)
    index = np.clip(index, 1, len(ks) - 1)
    left = ks[index - 1]
    right = ks[index]
    index -= freq - left < right - freq
    return index

# Given a spectrogram and threshold value, returns the list of note names that have a amplitude
# larger than the threshold, inserting empty strings where no note is found
def getNotes(spectrogram, ks, threshold):
    max_freq_ind = np.argmax(spectrogram, axis=0)
    max_freq = np.max(spectrogram, axis=0)
    max_freq_ind[max_freq < threshold] = 0
    note_frequencies = ks[max_freq_ind]
    notes = np.array([freq2note(frequency) for frequency in note_frequencies])
    return notes, note_frequencies

# Discards parts of the spectrogram outside of the given min and max frequencies
def clipSpectrogram(spectrogram, ks, min_freq, max_freq):
    min_freq_ind = getFreqIndex(ks, min_freq)
    max_freq_ind = getFreqIndex(ks, max_freq)

    spectrogram = spectrogram[min_freq_ind:max_freq_ind, :]
    ks = ks[min_freq_ind:max_freq_ind]

    return spectrogram, ks

# Takes the song matrices imported from matlab and returns the song vector and sample rate.
def getSongMatData(song):
    y = song['y'].flatten()
    Fs = song['Fs'].flatten()[0]
    return y, Fs


# Takes song vector and sample rate and returns the total number of samples, 
# length in seconds, time vector, and shifted frequency vector
def getSongData(y, Fs):
    n = len(y)
    song_length = n / Fs
    t = np.arange(0, n) / Fs
    k = np.append(np.arange(0, n/2), np.arange(-n/2, 0)) / song_length
    ks = fftshift(k)
    return n, song_length, t, ks

# Creates a spectrogram by first splitting it into parts and then stitching those partial
# spectrograms together
def createSpectrogramSplit(y, Fs, num_parts, num_windows, filter_width, log_transform, debug_plot = False):
    y_parts = np.split(y, num_parts)
    spectrogram = np.zeros((len(y_parts[0]), 0))
    all_taus = np.array([])
    for i, yi in enumerate(y_parts):
        print("Part: " + str(i + 1) + " / " + str(num_parts))

        _, song_length, _, _ = getSongData(yi, Fs)

        spectrogram_partial, taus, ks = createSpectrogram(yi, Fs, num_windows, filter_width, log_transform, debug_plot)
        all_taus = np.hstack((all_taus, taus + song_length * i))

        spectrogram = np.hstack((spectrogram, spectrogram_partial))

    return spectrogram, all_taus, ks

# Creates a spectrogram of a given song vector
def createSpectrogram(y, Fs, num_windows, filter_width, log_transform, debug_plot = False):
    n, _, t, ks = getSongData(y, Fs)

    taus = np.linspace(0, t[-1], num_windows)
    spectrogram = np.zeros((n, num_windows))

    for i, tau in enumerate(taus):
        print("Progress: " + str(i + 1) + " / " + str(num_windows))
        gauss = np.exp(-filter_width * (t - tau)**2)
        window = y * gauss
        y_transform = fftshift(fft(window))

        if log_transform:
            spectrogram[:, i] = np.log(np.abs(y_transform) + 1)
        else:
            spectrogram[:, i] = np.abs(y_transform)

        if debug_plot and i == int(num_windows / 4):
            plt.plot(t, gauss)
            plt.show()

            plt.plot(t, y * gauss)
            plt.show()

            plt.plot(ks, y_transform)
            plt.show()

    return spectrogram, taus, ks

# Plots a spectrogram
def plotSpectrogram(spectrogram, taus, ks, title=""):
    plt.pcolormesh(taus, ks, spectrogram, shading='gourad', cmap='hot')
    plt.colorbar()
    plt.title(title, size=28)
    plt.xlabel("Time", size=24)
    plt.ylabel("Frequency", size=24)

# Adds text annotations of note names to a spectrogram
def labelNotes(notes, taus, note_frequencies):
    prev_note = ""
    for i, note in enumerate(notes):
        if note != prev_note:
            note_txt = note
            if len(note) == 3:
                note_txt = "$\mathregular{" + note[0] + "^\\" + note[1] + "_" + note[2] + "}$"
            elif len(note) == 2:
                note_txt = "$\mathregular{" + note[0] + "_" + note[1] + "}$"
            plt.annotate(note_txt, (taus[i], note_frequencies[i]), color='b', weight='bold', fontsize=22)
        prev_note = note

# Plots a spectrogram with note names added
def plotSpectrogramWithNotes(spectrogram, taus, ks, threshold, title=""):
    plotSpectrogram(spectrogram, taus, ks, title)
    notes, note_frequencies = getNotes(spectrogram, ks, threshold)
    labelNotes(notes, taus, note_frequencies)
    plt.show()

# Load data from MATLAB matrices
gnr = sp.loadmat('GNR.mat')
floyd = sp.loadmat('Floyd.mat')

## Part 1
# Create Spectrogram and Label Notes for GNR
y, Fs = getSongMatData(gnr)

spectrogram, taus, ks = createSpectrogram(y, Fs, num_windows=100, filter_width=500, log_transform=True)

# Normalize spectrogram
spectrogram = spectrogram / np.amax(spectrogram)

# Throw away values outside normal music range for performance reasons
spectrogram, ks = clipSpectrogram(spectrogram, ks, min_freq=0, max_freq=1000)

plotSpectrogramWithNotes(spectrogram, taus, ks, threshold=.3, title="Spectrogram of Guns N' Roses Sample")


# Create Spectrogram and Label Notes for Floyd
y, Fs = getSongMatData(floyd)
y = y[0:-1]

spectrogram, taus, ks = createSpectrogramSplit(y, Fs, num_parts=10, num_windows=25, filter_width = 50, log_transform=True)

# Normalize spectrogram, crop to desired area
spectrogram = spectrogram / np.amax(spectrogram)
spectrogram, ks = clipSpectrogram(spectrogram, ks, min_freq=0, max_freq=1000)

plotSpectrogramWithNotes(spectrogram, taus, ks, threshold=.5, title="Spectrogram of Pink Floyd Sample")


## Part 2
y, Fs = getSongMatData(floyd)
y = y[0:-1]

n, _, _, ks = getSongData(y, Fs)

# Take Fourier Transform and apply a band pass filter to isolate the bass
y_transform = fftshift(fft(y))
lf = 60
hf = 150
band_filter = np.zeros(n)
band_filter[getFreqIndex(ks, lf):getFreqIndex(ks, hf)] = 1

y_transform_filtered = y_transform * band_filter
y_filtered = ifft(ifftshift(y_transform_filtered))

# Now create the spectrogram as above, except using the filtered version
spectrogram_bass, taus, ks = createSpectrogramSplit(y_filtered, Fs, num_parts=10, num_windows=25, filter_width=5, log_transform=True)

# Normalize spectrogram, crop to desired area
spectrogram_bass = spectrogram_bass / np.amax(spectrogram_bass)
spectrogram_bass, ks = clipSpectrogram(spectrogram_bass, ks, min_freq=50, max_freq=160)

plotSpectrogramWithNotes(spectrogram_bass, taus, ks, threshold=.5, title="Spectrogram of Isolated Bass Guitar from Pink Floyd Sample")

## Part 3
y, Fs = getSongMatData(floyd)
y = y[0:-1]
n, _, _, ks = getSongData(y, Fs)

y_transform = fftshift(fft(y))
lf = 150
hf = 20000
band_filter = np.zeros(n)
band_filter[getFreqIndex(ks, lf):getFreqIndex(ks, hf)] = 1

y_transform_filtered = y_transform * band_filter
y_filtered = ifft(ifftshift(y_transform_filtered))

# Now create the spectrogram as above, except using the filtered version
spectrogram, taus, ks = createSpectrogramSplit(y_filtered, Fs, num_parts=10, num_windows=25, filter_width=5, log_transform=True)

# Normalize spectrogram, crop to desired area
spectrogram = spectrogram / np.amax(spectrogram)
spectrogram, ks = clipSpectrogram(spectrogram, ks, min_freq=0, max_freq=1200)

# Get the bass note names
bass_notes, _ = getNotes(spectrogram_bass, ks, threshold=.5)

overtone_filters = np.zeros(spectrogram.shape)

for i, note in enumerate(bass_notes):
    # Find the frequency of each note to construct the overtones from
    freq = note2freq(note)

    num_overtones = 15
    gauss_filters = np.zeros((spectrogram.shape[0], num_overtones))
    # Construct a Gaussian filter for each overtone
    for overtone_num in range(num_overtones):
        filter_width = .005
        center_frequency = freq * (overtone_num + 1)
        gauss_filters[:, overtone_num] = np.exp(-filter_width* (ks - center_frequency)**2) / (overtone_num + 1)
    # Add up all the Gaussian filters and invert it
    combined_filter = 1 - np.sum(gauss_filters, axis=1)

    # Add the filter for each time point
    overtone_filters[:, i] = combined_filter

# Apply the filter to the spectrogram
spectrogram_no_overtones = spectrogram * overtone_filters

plotSpectrogram(overtone_filters, taus, ks, title="Spectrogram of Filter to Remove Overtones")
plt.show()

plotSpectrogramWithNotes(spectrogram_no_overtones, taus, ks, threshold=.25, title="Spectrogram of Pink Floyd Sample w/o Bass Guitar, Filtered to Remove Overtones")

plotSpectrogramWithNotes(spectrogram, taus, ks, threshold=.5, title="Spectrogram of Pink Floyd Sample w/o Bass Guitar, No Filter Applied")