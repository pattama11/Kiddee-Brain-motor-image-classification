import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from tqdm import tqdm
from scipy.signal import iirnotch, filtfilt, butter
import pywt
from functools import partial
from pywt import wavedec, swt, waverec, threshold
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def normalization(data: List[Union[pd.DataFrame, np.ndarray]], chan_names: List[str], per_channel: bool = True, clamp: float = 20.0) -> List[Union[pd.DataFrame, np.ndarray]]:
    if per_channel:
        get_mean = partial(np.mean, axis=0, keepdims=True)
        get_std = partial(np.std, axis=0, keepdims=True)
    else:
        get_mean = np.mean
        get_std = np.std

    for i in tqdm(range(len(data)), total=len(data), desc="Normalization"):
        if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
            raise TypeError('Data must be of type pd.DataFrame or np.ndarray')
        
        if isinstance(data[i], pd.DataFrame): #for dataframes
            channel_data = data[i][chan_names].values
            mean = get_mean(channel_data)
            std = get_std(channel_data)
            std[std == 0] = 1  # Avoid division by zero

            # Normalize data in-place
            np.subtract(channel_data, mean, out=channel_data)
            np.divide(channel_data, std, out=channel_data)

            if clamp:
                np.clip(channel_data, a_min=-clamp, a_max=clamp, out=channel_data)
            # Replace original channel columns with normalized data
            data[i][chan_names] = channel_data

        else: #for arrays
            mean = get_mean(data[i])
            std = get_std(data[i])
            std[std == 0] = 1  # Avoid division by zero

            # Normalize data in-place
            np.subtract(data[i], mean, out=data[i])
            np.divide(data[i], std, out=data[i])
            if clamp:
                np.clip(data[i], a_min=-clamp, a_max=clamp, out=data[i])
    return data

def notch_filter(data: List[Union[pd.DataFrame, np.ndarray]], chan_names: List[str], fs: float, freqs: List[float] = [60.0]) -> List[Union[pd.DataFrame, np.ndarray]]:
    if isinstance(freqs, float):
        freqs = [freqs]
    assert len(freqs) > 0, "frequency must be a list of floats"

    for i in tqdm(range(len(data)), total=len(data), desc="Notch filtering"):
        if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
            raise TypeError('Data must be of type pd.DataFrame or np.ndarray')

        for f in freqs:
            b, a = iirnotch(f, 30.0, fs=fs)
            if isinstance(data[i], pd.DataFrame):
                channel_data = data[i][chan_names]
                channel_data = filtfilt(b, a, channel_data, axis=0) 
                data[i][chan_names] = channel_data
            else:
                data[i] = filtfilt(b, a, data[i], axis=0)
    return data

def bandpass_filter(data: List[Union[pd.DataFrame, np.ndarray]], chan_names: List[str], fs: float, low: Optional[float] = 1.0, high: Optional[float] = 95.0, order: Optional[int] = 4) -> List[Union[pd.DataFrame, np.ndarray]]:
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq

    # Design the bandpass filter using Butterworth filter
    b, a = butter(order, [low_norm, high_norm], btype='bandpass')

    for i in tqdm(range(len(data)), total=len(data), desc="Bandpass filtering"):
        if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
            raise TypeError('Data must be of type pd.DataFrame or np.ndarray')
        
        if isinstance(data[i], pd.DataFrame): #for dataframes
            channel_data = data[i][chan_names]
            channel_data = filtfilt(b, a, channel_data, axis=0)
            data[i][chan_names] = channel_data
        else: #for arrays
            data[i] = filtfilt(b, a, data[i], axis=0)
    return data

def dwt_denoising(data: List[Union[pd.DataFrame, np.ndarray]], chan_names: List[str], type="dwt", wavelet="db8", mode="sym", level: int = 4, method: Union[str, float] = "soft") -> List[Union[pd.DataFrame, np.ndarray]]:
    types = ["dwt", "swt"]
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)
    assert method in ["soft", "hard", "garrote", "greater", "less"] or isinstance(method, float), "Invalid method. Expected one of: [soft, hard, garrote, greater, less] or of type float"
    for i in tqdm(range(len(data)), total=len(data), desc="DWT denoising"):
        if isinstance(data[i], pd.DataFrame): #if dataframe
            channel_data = data[i][chan_names].values
            denoised_data = _denoise(channel_data, type, wavelet, mode, level, method)
            data[i][chan_names] = denoised_data
        elif isinstance(data[i], np.ndarray): #if ndarray
            denoised_data = _denoise(data[i], type, wavelet, mode, level, method)
            data[i] = denoised_data
        else:
            raise TypeError('Data type not understood. Please provide a DataFrame or ndarray.')
    return data

def fourier_transform(data: List[Union[pd.DataFrame, np.ndarray]], chan_names: List[str]) -> List[Union[pd.DataFrame, np.ndarray]]:
    for i in tqdm(range(len(data)), total=len(data), desc="Fourier Transform"):
        if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
            raise TypeError('Data must be of type pd.DataFrame or np.ndarray')
        
        if isinstance(data[i], pd.DataFrame):
            channel_data = data[i][chan_names].values
            transformed_data = np.fft.fft(channel_data, axis=0)
            data[i][chan_names] = transformed_data
        else:
            transformed_data = np.fft.fft(data[i], axis=0)
            data[i] = transformed_data
    
    return data

def _denoise(channel_data, type, wavelet, mode, level, method):
    if type == "dwt":
        transform = partial(wavedec, wavelet=wavelet, level=level, axis=0)
    else:
        transform = partial(swt, wavelet=wavelet, level=level, axis=0, trim_approx=True)
    coeffs = transform(channel_data)

    # Simple Threshold Denoising
    if isinstance(method, str):
        thresholding = partial(threshold, mode=method, substitute=0)
        for j in range(len(coeffs[1:])):
            sig = np.median(np.abs(coeffs[j+1]), axis=0) / 0.6745
            thresh = sig * np.sqrt(2 * np.log(len(coeffs[j+1])))
            if np.all(thresh) == True:
                thresholded_detail = thresholding(data=coeffs[j+1], value=thresh)
                coeffs[j+1] = thresholded_detail
        channel_data = waverec(coeffs, wavelet=wavelet, axis=0)

    # Neigh-Block Denoising (To do: make more efficient with vectorization)
    else: #Method used as "sigma"
        # helper function to compute beta shrinkage
        def beta(method, L, detail, lmbd):
            S2 = np.sum(detail ** 2)
            beta = (1 - lmbd * L * method**2 / S2)
            return max(0, beta)

        L0 = int(np.log2(len(channel_data)) // 2)
        L1 = max(1, L0 // 2)
        L = L0 + 2 * L1
        lmbd = 4.50524  # explanation in Cai & Silverman (2001)
        for j, detail in enumerate(coeffs[1:]):
            d2 = detail.copy()
            # group wavelet into disjoint blocks of length L0
            for start_b in range(0, len(d2), L0):
                end_b = min(len(d2), start_b + L0)  # if len(d2) < start_b + L0 -> last block
                start_B = start_b - L1
                end_B = start_B + L
                if start_B < 0:
                    end_B -= start_B
                    start_B = 0
                elif end_B > len(d2):
                    start_B -= end_B - len(d2)
                    end_B = len(d2)
                assert end_B - start_B == L  # sanity check
                d2[start_b:end_b] *= beta(method, L, d2[start_B:end_B], lmbd)
        channel_data = waverec(coeffs, wavelet=wavelet, axis=0)
    return channel_data

# Set the path to the "test" folder
#test_folder = 'test'

def load_test_data(test_folder: str) -> List[np.ndarray]:
    predictions = []
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.endswith('.npy'):
                
                # Load data and labels for the current subject, session, part, and trial
                data = np.load(os.path.join(root, file))
                # Extract the main data segments (3 seconds) based on label timestamps
                segment_duration = 3  # Duration of the main data segment in seconds
                skip_duration = 2  # Duration of data to skip before and after the main segment
                sample_rate = 250  # Sampling rate of the EEG data
                segment_samples = segment_duration * sample_rate
            
                start_index = 500  # Skip the data before the main segment
                #start_index += skip_samples  # Skip the data before the main segment
                
                data_segments = data[:, :8]  # Extract the first 8 channels
                predictions.append(data_segments)
    
    return predictions




def pred(test_data):
    # Set the path to the best model file
    best_model_path = r'C:\Anacoda_En\SuperAI\web_last\Backend\best_model.keras'
    # Set the path to the submission file
    # submission_file = 'sample_submission.csv'

    # Load the best model
    best_model = tf.keras.models.load_model(best_model_path)

    # Initialize a list to store the predictions
    

    # Define the segment duration, skip duration, and sampling rate
    segment_duration = 3  # Duration of the main data segment in seconds
    skip_duration = 2  # Duration of data to skip before and after the main segment
    sample_rate = 250  # Sampling rate of the EEG data
    segment_samples = segment_duration * sample_rate
    skip_samples = skip_duration * sample_rate

    # Loop through each segment ID in the submission file

    # Select the first 8 columns (EEG channels)
    test_data = test_data[:, :8]

    # Skip the first 2 seconds and the last 2 seconds of data
    start_index = skip_samples
    end_index = start_index + segment_samples
    test_data = test_data[start_index:end_index]


        
        # Preprocess the test data
    chan_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']  # Replace with your channel names
    fs = 250  # Replace with your sampling frequency
    test_data = normalization([test_data], chan_names)
    test_data = notch_filter(test_data, chan_names, fs,freqs=[60])
    test_data = bandpass_filter(test_data, chan_names, fs, low=10.0, high=22.0, order=4)

    test_data = np.expand_dims(test_data[0], axis=0)
    #print(test_data.shape)
    #print(best_model)
    # Make predictions
    pred_prob = best_model.predict(test_data)
    
    pred_label = np.argmax(pred_prob, axis=1)[0]

    # Map the predicted label back to the original label value
    label_map_inv = {0: 110, 1: 120, 2: 150}
    pred_label = label_map_inv[pred_label]
    #predictions.append(pred_label)

    #print(pred_label)
    #print("Inference completed. Submission file 'submission.csv' updated with predictions.")
    return pred_label
    
if __name__ == '__main__':
    path = r"C:\Anacoda_En\SuperAI\web_last\Backend\test_application\test_application"
    if path.endswith('.npy'):
        brain = np.load()
        answer = pred(brain)
    else:
        answers = []
        many_list = load_test_data(path)
        for brain in many_list:
            answer = pred(brain)
            answers.append(answer)
    print(answers)