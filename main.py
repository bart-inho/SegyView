from scipy.fftpack import fft
from scipy.stats import skew, kurtosis
import tkinter as tk
from tkinter import filedialog
from obspy.io.segy.segy import _read_segy
from obspy.core.stream import Stream
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import gridspec
from tqdm import tqdm
import numpy as np

class ProfilePlotter:

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

    def detect_ringing(self, trace, window_size, threshold):
        """Detect ringing in a trace using a sliding window and return the detected positions."""
        half_window = window_size // 2
        detected_positions = []
        
        for i in range(half_window, len(trace) - half_window):
            window_sum = np.sum(np.abs(trace[i - half_window: i + half_window]))
            
            if window_sum > threshold:
                detected_positions.append(i)
                
        return detected_positions

    def quantify_ringing(self, trace, window_size, threshold):
        """Quantify the ringing in a trace and return an aggregate measure."""
        detected_positions = self.detect_ringing(trace, window_size, threshold)
        
        # Compute total amplitude within detected ringing regions
        aggregate_amplitude = sum([trace[pos] for pos in detected_positions])
        
        return aggregate_amplitude

    def plot_radargrams(self, profiles):
        """Plot the given profiles with indications of minimum amplitude locations."""
        fig, axs = plt.subplots(len(profiles), 1, figsize=(10, 5 * len(profiles)))
        
        if len(profiles) == 1:
            axs = [axs]

        ringing_quantifications = []

        for idx, profile in enumerate(profiles):
            
            threshold = 8e4

            ringing_amplitudes = [self.quantify_ringing(trace, 10, threshold) for trace in profile]  # Example values for window_size and threshold
            ringing_quantifications.append(ringing_amplitudes)
            
            im = axs[idx].imshow(profile.T, aspect='auto', cmap='seismic',
                             extent=[0, profile.shape[0], profile.shape[1], 0])
            axs[idx].set_title(f"Radargram {idx+1}")
            axs[idx].set_xlabel("Traces")
            axs[idx].set_ylabel("Time samples")

            # Add colorbar in scientific notation
            cbar = fig.colorbar(im, ax=axs[idx], format='%.1e')
            cbar.set_label('Amplitude')

        # Compare the ringing quantifications
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, ringing in enumerate(ringing_quantifications):
            ax.plot(ringing, label=f"Radargram {idx+1}")
            
        ax.set_title("Ringing Quantification Comparison")
        ax.set_xlabel("Traces")
        ax.set_ylabel("Aggregate Amplitude of Ringing")
        ax.legend()

        # Set y-axis to scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        
        # plt.tight_layout()
        # plt.show()

    def run(self):
        """Main execution function."""
        file_paths = self.select_files()

        self.profiles = []  # Initialize profiles as an empty list
        for file_path in file_paths:
            profile = self.read_sgy_file(file_path)
            self.profiles.append(profile)

        # if self.profiles:
        #     self.plot_radargrams(self.profiles)

class Analysis:

    def __init__(self, profiles):
        self.profiles = profiles
    
    def mean_amplitude(self, trace):
        return np.mean(np.abs(trace))

    def coherency(self, profile, window_size=3):
        """
        Compute the coherency of each trace with its adjacent traces.
        
        Args:
        profile (np.ndarray): 2D array with traces as columns.
        window_size (int): The size of the sliding window to compute coherency.
        
        Returns:
        np.ndarray: 2D array of coherency values.
        """
        # Number of traces
        num_traces = profile.shape[1]
        # Initialize coherency matrix with zeros
        coherency_matrix = np.zeros((profile.shape[0], num_traces))

        # Iterate over each trace, except the first and last which cannot have coherency
        for i in range(1, num_traces - 1):
            # Extract the main trace and its neighbors
            main_trace = profile[:, i]
            prev_trace = profile[:, i - 1]
            next_trace = profile[:, i + 1]

            # Compute coherency using a sliding window
            for j in range(window_size, len(main_trace) - window_size):
                window_main = main_trace[j - window_size:j + window_size]
                window_prev = prev_trace[j - window_size:j + window_size]
                window_next = next_trace[j - window_size:j + window_size]

                # Compute the average coherency with previous and next trace
                coherency_matrix[j, i] = (np.corrcoef(window_main, window_prev)[0, 1] +
                                          np.corrcoef(window_main, window_next)[0, 1]) / 2.0

        return coherency_matrix
    
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

    def SubPlotAnalysis(self, profiles, amplitude, max_trace, max_mean_index, min_trace, min_mean_index, index=-1):
        
        fig = plt.figure(figsize=(10, 15))  # Adjust the overall figure size as needed
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05])  # Adjust the width ratio for colorbars

        # Energy plot
        ax0 = plt.subplot(gs[0, 0])
        ax0.plot(amplitude, 'k')
        ax0.set_title("Mean trace amplitude")
        ax0.set_xlabel("Traces")
        ax0.set_ylabel("Amplitude [V/m]")
        ax0.grid(which='major', axis='both', linestyle='--', color='k', linewidth=.1)

        # Radargram plot with colorbar
        ax4 = plt.subplot(gs[2, 0])
        im4 = ax4.imshow(profiles.T, aspect='auto', cmap='seismic')
        ax4.set_title("Radar Profile")
        ax4.set_xlabel("Traces")
        ax4.set_ylabel("Time samples [ns]")
        cbar4 = plt.colorbar(im4, cax=plt.subplot(gs[2, 1]))
        cbar4.formatter.set_powerlimits((0, 0))  # Use scientific notation
        cbar4.update_ticks()

        # Plot for max and min mean amplitude trace comparison
        ax_compare = plt.subplot(gs[1, 0])
        ax_compare.plot(max_trace, 'b', linewidth = .8 , label='Max - Trace nb = '+str(max_mean_index))
        ax_compare.plot(min_trace, 'r', linewidth = 1.5 , label='Min - Trace nb = '+str(min_mean_index))
        ax_compare.set_title("Comparison of Max and Min Mean Amplitude Traces")
        ax_compare.set_xlabel("Samples")
        ax_compare.set_ylabel("Amplitude [V/m]")
        ax_compare.legend()
        ax_compare.grid(which='major', axis='both', linestyle='--', color='k', linewidth=.1)

        plt.tight_layout()
        plt.savefig(f'figures_proc4/profile_analysis_{index}.pdf')  # Save each figure with its index number
        plt.close(fig)

    def process_profiles(self):
        results = []

        # Wrap the enumerator with tqdm for a progress bar
        for index, profile in tqdm(enumerate(self.profiles), total=len(self.profiles), desc="Processing Profiles"):

            n_traces_for_background = 20  # Example value, adjust as needed
            profile = self.background_removal(profile, n_traces_for_background)

            if index == 2:
                profile = profile[34:, :]

            max_mean_index, min_mean_index = self.find_extreme_mean_traces(profile)
            max_trace = profile[max_mean_index, :]
            min_trace = profile[min_mean_index, :]
            
            amplitude = [self.mean_amplitude(trace) for trace in profile]
            # coherency = self.coherency(profile, window_size=5)

            self.SubPlotAnalysis(profile, amplitude, max_trace, max_mean_index, min_trace, min_mean_index, index)

        return results

if __name__ == "__main__":
    plotter = ProfilePlotter()
    plotter.run()  # This will read the files and populate the profiles attribute

    if hasattr(plotter, 'profiles') and plotter.profiles:
        analysis = Analysis(plotter.profiles)
        results = analysis.process_profiles()
    else:
        print("No profiles to process.")
