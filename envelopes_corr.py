import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn

#### Параметры, которые нужно поменять
name_raw_channel = 'EEG-0'#'P3'
name_channel_with_envelope = 'EEG-1'#'Cz'
pathdir = 'C:/Users/Fedosov/Downloads/BCICIV_2a_gdf/A01T.gdf'
####


raw = mne.io.read_raw_gdf(pathdir)
fs = int(round(raw.info['sfreq']))

raw_data = raw[name_raw_channel][0][0]
envelope_on_board = raw[name_channel_with_envelope][0][0]

search_area = int(fs*2)
corr_store = np.zeros(np.min([search_area*2, raw_data.shape[0]]))

b, a = sgn.butter(3, btype='bandpass', Wn=[8, 12], fs=fs)


filtered = sgn.filtfilt(b,a,raw_data)
envelope_gt = np.abs(sgn.hilbert(filtered))

for i, shift in enumerate(np.arange(-search_area, search_area)):
    corr_store[i] = np.corrcoef(np.roll(envelope_gt, shift), envelope_on_board)[0,1]

corr = np.max(corr_store)

print(corr)

plt.figure()
plt.plot(envelope_gt[10*fs:15*fs])
plt.plot(envelope_on_board[10*fs:15*fs])
plt.show()
