import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import tkinter as tk
from tkinter import filedialog
from obspy.io.segy.segy import _read_segy
from scipy.signal import butter, filtfilt
from tqdm import tqdm


class ReadSegy:

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

    def select_files(self):
        """Open a file dialog and return the selected file paths."""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("SGY files", ".sgy")])
        return file_paths

    def read_sgy_file(self, file_path):
        """Read the given .sgy file and return the data."""
        stream = _read_segy(file_path)
        data = np.stack([trace.data for trace in stream.traces])
        return data

    def run(self):
        """Main execution function."""
        file_paths = self.select_files()

        self.profiles = []  # Initialize profiles as an empty list
        for file_path in file_paths:
            profile = self.read_sgy_file(file_path)
            self.profiles.append(profile)

class Analysis:

    def __init__(self, profiles):
        self.profiles = profiles
    
    def mean_amplitude(self, trace):
        return np.mean(np.abs(trace))
    
    def find_extreme_mean_traces(self, profile):
        """
        Find the traces with the maximum and minimum mean amplitude in the profile.
        
        Args:
        profile (np.ndarray): 2D array with traces as columns.
        
        Returns:
        tuple: Indices of the trace with the max mean amplitude and the trace with the min mean amplitude.
        """
        # Calculate the mean amplitude for each trace
        mean_amplitudes = np.mean(np.abs(profile), axis=1)
        max_mean_index = np.argmax(mean_amplitudes)
        min_mean_index = np.argmin(mean_amplitudes)

        return max_mean_index, min_mean_index

    def set_time_zero(self, profile, zero_time=400):
        """
        Set time to zero
        """
        point_per_trace = 1250
        total_time_window = 4000
        zero_index = int(zero_time / total_time_window * point_per_trace)
        profile = profile[:, zero_index:]

        return profile

    def time_window_cut(self, profile, time_cut=750):
        """
        Cuts the time window of the profile to expose only the n first nanoseconds
        """
        
        # Calculate the index to cut based on the time in nanoseconds
        point_per_trace = 1250
        total_time_window = 4000
        time_index = int(time_cut / total_time_window * point_per_trace)

        profile = profile[:, :time_index]

        return profile

    def background_removal(self, profile, n_traces):
        """ 
        Remove background noise from the profile by averaging the first n traces and 
        subtracting this average from each trace in the profile.
        
        Args:
        profile (np.ndarray): 2D array with traces as columns.
        n_traces (int): Number of traces to use for background averaging.
        
        Returns:
        np.ndarray: Profile with background noise removed.
        """
        # Calculate the mean of the first n traces
        background_mean = profile[:n_traces, :].mean(axis=0)

        # Subtract the background mean from each trace
        profile -= background_mean

        return profile
    
    def butterworth_bandpass(self, profile, lowcut, highcut, fs, index, order=5):
        """
        Apply a Butterworth bandpass filter to the profile.

        Args:
        profile (np.ndarray): 2D array with traces as columns.
        lowcut (float): Low cutoff frequency of the filter in Hz.
        highcut (float): High cutoff frequency of the filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): The order of the filter.

        Returns:
        np.ndarray: The filtered profile.
        """

        # Nyquist Frequency
        nyq = 0.5 * fs

        # Normalized frequencies must be in the range 0 to 1
        low = lowcut / nyq
        high = highcut / nyq

        # Design the Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')

        # Apply the filter to each trace in the profile
        filtered_profile = np.zeros_like(profile)
        for i in range(profile.shape[0]):
            filtered_profile[i,:] = filtfilt(b, a, profile[i,:])

        return profile, filtered_profile

    def SubPlotAnalysis(self, 
                        profile,
                        backg_profile, 
                        max_trace, 
                        max_mean_index, 
                        max_mean_index_control,
                        max_trace_control,
                        index=-1):

        fig = plt.figure(figsize=(7, 8), constrained_layout=True)
        axs = fig.subplot_mosaic([
            ["profile", "spectrum"],
            ["comparison", "comparison"],
            ["difference", "difference"]
        ])

        plt.suptitle(f"Profile Analysis LINE{index}")

        # Calculate the time values for the y-axis
        sampling_frequency = 1250  # points per trace
        time_window = 4000e-9  # seconds
        time_in_ns = np.arange(profile.shape[1]) * (time_window * 1e9) / sampling_frequency
        # Plot the radar profile using pcolor
        profile_img = axs['profile'].pcolor(np.arange(profile.shape[0]), time_in_ns, profile.T, cmap='seismic')
        axs['profile'].set_title("Radar Profile")
        axs['profile'].set_xlabel("Traces")
        axs['profile'].set_ylabel("Time [ns]")
        axs['profile'].invert_yaxis()

        # Add colorbar for the radar profile
        cbar = fig.colorbar(profile_img, ax=axs['profile'], location='left', pad=-0.15)

        # Create a ScalarFormatter object
        formatter = ScalarFormatter(useMathText=True)  # This enables scientific notation
        formatter.set_scientific(True)  # Always use scientific notation
        formatter.set_powerlimits((0,0))  # This will force scientific notation
        # Apply the formatter to the colorbar
        cbar.ax.yaxis.set_major_formatter(formatter)

        # Plot the amplitude spectrum
        axs['spectrum'].magnitude_spectrum(np.mean(backg_profile, axis=0), color='.6', scale='dB', Fs=1250/4000e-9, label='RAW')
        axs['spectrum'].magnitude_spectrum(np.mean(profile, axis=0), color='r', scale='dB', Fs=1250/4000e-9, label='Filtered')
        axs['spectrum'].set_title("Amplitude Spectrum")
        axs['spectrum'].set_xlabel("Frequency [MHz]")
        axs['spectrum'].set_ylabel("Amplitude [dB]")
        # add a vertical line at 25 MHz
        axs['spectrum'].axvline(x=25e6, color='.5', linestyle='--', linewidth=.5)
        axs['spectrum'].set_ylim(bottom=0)
        axs['spectrum'].set_xlim(left=0)
        axs['spectrum'].set_xlim(right=150e6)
        axs['spectrum'].legend()
        locs = axs['spectrum'].get_xticks()
        axs['spectrum'].set_xticks(locs, map(lambda x: "{:.0f}".format(x/1e6), locs))

        # Plot for max mean amplitude trace comparison
        axs['comparison'].plot(time_in_ns, max_trace_control, 'k', label='Control - Trace nb = '+str(max_mean_index_control))
        axs['comparison'].plot(time_in_ns, max_trace, 'r', label='Max - Trace nb = '+str(max_mean_index))
        axs['comparison'].set_title("Comparison of Max Mean Amplitude Traces")
        axs['comparison'].set_xlabel("Samples")
        axs['comparison'].set_ylabel("Amplitude [V/m]")
        axs['comparison'].set_xlim(left=0)
        axs['comparison'].legend()
        axs['comparison'].grid(which='major', axis='both', linestyle='--', color='k', linewidth=.1)

        # Plot the difference between the max mean amplitude trace and the control trace
        axs['difference'].plot(time_in_ns, np.abs(max_trace - max_trace_control), 'k', label='Trace Substraction')
        axs['difference'].set_title("Control - Shielding")
        axs['difference'].set_xlabel("Samples")
        axs['difference'].set_ylabel("Amplitude [V/m]")
        axs['difference'].set_xlim(left=0)
        axs['difference'].legend()
        axs['difference'].grid(which='major', axis='both', linestyle='--', color='k', linewidth=.1)

        plt.savefig(f'figures/profile_analysis_{index}.pdf')  # Save each figure with its index number
        plt.close(fig)

    def process_profiles(self):

        # Wrap the enumerator with tqdm for a progress bar
        for index, profile in tqdm(enumerate(self.profiles), total=len(self.profiles), desc="Processing Profiles"):

            # GROUNDED DATA
            if index == 0:
                profile = profile[150:-20, :]

            if index == 2:
                profile = profile[20:-20, :]   

            if index == 3:
                profile = profile[80:-20, :]

            if index == 5:
                profile = profile[20:180, :]
            
            # NOT GROUNDED DATA

            # if index == 0:
            #     profile = profile[:200, :]

            # if index == 2:
            #     profile = profile[34:, :]

            # if index == 4:
            #     profile = profile[125:, :]

            # if index == 13:
            #     profile = profile[:190, :]

            zero_time = 420
            profile = self.set_time_zero(profile, zero_time)

            time_cut = 750
            profile = self.time_window_cut(profile, time_cut)

            n_traces_for_background = 10  # Example value, adjust as needed
            profile = self.background_removal(profile, n_traces_for_background)

            # Apply Butterworth Bandpass filter
            fs = 1250/4000e-9  # Example sampling frequency in Hz (adjust as needed)
            backg_profile, profile = self.butterworth_bandpass(profile, 10e6, 60e6, fs, index)

            if index == 0:
                max_mean_index_control, min_mean_index_control = self.find_extreme_mean_traces(profile)
                max_trace_control = profile[max_mean_index_control, :]
                min_trace_control = profile[min_mean_index_control, :]

            max_mean_index, min_mean_index = self.find_extreme_mean_traces(profile)
            max_trace = profile[max_mean_index, :]
            min_trace = profile[min_mean_index, :]

            self.SubPlotAnalysis(profile, 
                                 backg_profile, 
                                 max_trace, 
                                 max_mean_index, 
                                 max_mean_index_control,
                                 max_trace_control,
                                 index)

if __name__ == "__main__":
    read = ReadSegy()
    read.run()  # This will read the files and populate the profiles attribute

    if hasattr(read, 'profiles') and read.profiles:
        analysis = Analysis(read.profiles)
        analysis.process_profiles()
    else:
        print("No profiles to process.")