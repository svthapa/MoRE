import numpy as np 
import random
import numpy as np
import random
from scipy.interpolate import interp1d

def time_warp(segment, target_length):
    x = np.linspace(0, 1, len(segment))
    f = interp1d(x, segment, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

def process_ecg_channel(channel, m, w):
    segment_length = len(channel) // m
    modified_channel = []
    segments_to_modify = random.sample(range(m), m // 2)

    for i in range(m):
        segment = channel[i * segment_length: (i+1) * segment_length]
        target_length = segment_length + int(segment_length * w if i in segments_to_modify else -segment_length * w)
        
        # Adjust the last segment to ensure the total length is 5000
        if i == m - 1:
            target_length = 1000 - sum(len(s) for s in modified_channel)

        modified_segment = time_warp(segment, target_length)
        modified_channel.append(modified_segment)

    return np.concatenate(modified_channel)

def time_warp_ecg(ecg_data, m=4, w=0.25):
    if m % 2 != 0:
        raise ValueError("m must be an even number.")

    modified_ecg_data = np.array([process_ecg_channel(channel, m, w) for channel in ecg_data])
    return modified_ecg_data


def permutation_augmentation(ecg_signal, m=4):
    """
    Apply permutation augmentation on the given ECG signal, preserving channel order.

    Parameters:
    - ecg_signal: The input ECG signal with shape (num_channels, num_samples)
    - m: The number of segments to divide each channel into

    Returns:
    - Augmented signal with shape (num_channels, num_samples)
    """
    # Get the number of channels and samples

    num_channels, num_samples = ecg_signal.shape
    
    # Check if the signal length is divisible by 'm'
    if num_samples % m != 0:
        raise ValueError("Signal length is not divisible by 'm'")
    
    # Calculate the length of each segment within a channel
    segment_length = num_samples // m
    
    # Initialize an empty array for the augmented signal
    augmented_signal = np.empty_like(ecg_signal)
    
    # Divide each channel into 'm' segments, shuffle them, and concatenate back
    for channel in range(num_channels):
        channel_data = ecg_signal[channel, :]
        segments = [channel_data[i * segment_length : (i + 1) * segment_length] for i in range(m)]
        np.random.shuffle(segments)
        augmented_signal[channel, :] = np.concatenate(segments)
    
    return augmented_signal


def add_gaussian_noise(ecg_data, sigma):
    """
    Adds Gaussian noise to the ECG data.

    Parameters:
    - ecg_data: The ECG signal data.
    - sigma: The standard deviation of the Gaussian noise.

    Returns:
    - augmented: The ECG signal with added Gaussian noise.
    """
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(loc=0, scale=sigma, size=ecg_data.shape)
    
    # Add the noise to the ECG data
    augmented = ecg_data + noise
    
    return augmented

class ECGAugmentor:
    def __init__(self):
        pass
    
    def time_warp(self, signal, factor = 0.25, points = 4):
        return time_warp_ecg(signal, w = factor, m = points)
    
    def permutation(self, signal, m=4):
        return permutation_augmentation(signal, m=m)
    
    def add_noise(self, signal, sigma = 0.2):
        return add_gaussian_noise(signal, sigma=sigma)
    
    def jitter(self, ecg_data, noise_factor=0.03):
        noise = np.random.randn(*ecg_data.shape) * noise_factor
        augmented = ecg_data + noise
        return np.clip(augmented, -1.0, 1.0)
    
    def random_scaling(self, signal, scale_range=(0.8, 2.1)):
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale_factor
    
    def flip_signal(self, signal):
        if np.random.rand() > 0.5:
            return -signal
        return signal
    
    def no_aug(self, signal):
        return signal
        
    
    def randomAugment(self, signal, augmentations):
        selected_augments = random.sample(augmentations, 1)
        for augment in selected_augments:
            signal = augment(signal)
        return signal
    

    def augment(self, signal):
        signal = self.randomAugment(signal, augmentations = [
                                            self.time_warp, self.permutation, #self.add_noise, 
                                            self.no_aug
                                    ])
        return signal