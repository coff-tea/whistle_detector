"""
From https://github.com/coff-tea/whistle_detector
""" 


import random
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from scipy.signal import spectrogram as sspec
from scipy.signal import butter, lfilter



def split_sets(num_idx, splits, replicable=True, seed=42):
    """Split a set of indices into as many as indicated by splits dict, which are given in progressive
    decimals representing portion into each following set. Replicability based on random seed. Returns
    dictionary with index splits."""
    idx = [i for i in range(num_idx)]
    if replicable:
        random.seed(seed)
    idx_splits = []
    for i in range(len(splits)):
        random_state = random.randint(0, 2**32-1)
        retain, idx = train_test_split(idx, train_size=splits[i], random_state=random_state)
        idx_splits.append(retain)
    idx_splits.append(idx)
    return(idx_splits)


def load_data(X, sr=50000, rescale=True, window="hamming", nperseg=1024, noverlap=None, scaling="spectrum"):
    """Creates list of spectrograms from list of signals."""
    X_spec = []
    for i in range(len(X)):
        _, _, ssx = sspec(X[i], fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
        if rescale:
            ssx = 10*np.log10(ssx)
        ssx = (ssx-np.min(ssx)) / (np.max(ssx)-np.min(ssx))
        ssx = np.nan_to_num(ssx)
        X_spec.append(ssx)
    return X_spec


def process_data(X, idx, mode, mdl_chs, spat_dim, tag=None, which_ch=None):
    """Creates list of data samples with given specifications. Assume that:
        - single/avg/all mode: mdl_chs=1
        - stk mode: mdl_chs=chs
        - stkwavg mode: mdl_chs>chs
    """
    data = []
    chs = len(X)
    if mode == "all":
        for i in idx:
            for c in range(chs):
                sample = np.expand_dims(resize(X[ch][i], (spat_dim, spat_dim)), axis=0)
                if tag is not None:
                    data.append([sample, tag])
                else:
                    data.append([sample])
    else:
        for i in idx:
            sample = None
            if "stk" in mode:
                frames = []
                for ch in range(chs):
                    frames.append(resize(X[ch][i], (spat_dim, spat_dim)))
                if mode == "stkwavg":
                    avg = np.mean(np.array(frames), axis=0)
                    avg = np.nan_to_num((avg-np.min(avg)) / (np.max(avg)-np.min(avg)))
                    for extra in range(mdl_chs-chs):        # Repeat if necessary
                        frames.append(avg)
                    stack_chs = [ch for ch in range(len(frames)-1)]
                    stack_chs.insert(0, len(frames)-1)
                    sample = np.dstack(tuple(frames)).transpose(tuple(stack_chs))
                else:
                    sample = np.dstack(tuple(frames))
            elif mode == "single" and which_ch is not None:
                sample = np.expand_dims(resize(X[which_ch-1][i], (spat_dim, spat_dim)), axis=0)
            elif mode == "avg":
                frames = []
                for ch in range(chs):
                    frames.append(resize(X[ch][i], (spat_dim, spat_dim)))
                avg = np.mean(np.array(frames), axis=0)
                avg = np.nan_to_num((avg-np.min(avg)) / (np.max(avg)-np.min(avg)))
                sample = np.expand_dims(avg, axis=0)
            if sample is not None:
                if tag is not None:
                    data.append([sample, tag])
                else:
                    data.append([sample])
    return data


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


# Use this function if necessary
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_data_tf(X, sr=50000, rescale=True, window="blackman", nperseg=2048, noverlap=410, scaling="spectrum"):
    """Creates list of spectrograms from list of signals."""
    print("TF!")
    X_spec = []
    for i in range(len(X)):
        windowed = butter_bandpass_filter(X[i], 5000, 20000, sr)
        windowed = windowed[int(sr*0.1):int(sr*0.9)]
        _, _, ssx = sspec(windowed, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
        ssx = ssx[int(3/25*ssx.shape[0]):int(20/25*ssx.shape[0]), :]
        if rescale:
            ssx = 10*np.log10(ssx)
        ssx = (ssx-np.min(ssx)) / (np.max(ssx)-np.min(ssx))
        ssx = np.nan_to_num(ssx)
        X_spec.append(ssx)
    return X_spec


def process_data_tf(X, idx, stack, spat_dim, use_tf=True, tag=None):
    """Creates list of data samples to suit other model's specifications."""
    print("TF!")
    data = []
    chs = len(X)
    for i in idx:
        frames = []
        for ch in range(chs):
            frames.append(resize(X[ch][i], (spat_dim, spat_dim)))
        avg = np.mean(np.array(frames), axis=0)
        avg = (avg-np.min(avg)) / (np.max(avg)-np.min(avg))
        if stack == 1:
            sample = np.expand_dims(avg, axis=0)
        else:
            shaped = [avg for _ in range(stack)]
            stack_chs = [ch for ch in range(stack-1)]
            stack_chs.insert(0, (stack-1))
            sample = np.dstack(tuple(shaped)).transpose(tuple(stack_chs))
        if tag is not None:
            data.append([sample, tag])
        elif not use_tf:
            data.append([sample])
        else:
            data.append(sample)
    return data
