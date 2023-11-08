from scipy.fftpack import fft
from scipy.stats import skew, kurtosis
import tkinter as tk
from tkinter import filedialog
from obspy.io.segy.segy import _read_segy
from obspy.core.stream import Stream
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

class RadarGramPlotter:

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

    def plot_radargrams(self, radargrams):
        """Plot the given radargrams with indications of minimum amplitude locations."""
        fig, axs = plt.subplots(len(radargrams), 1, figsize=(10, 5 * len(radargrams)))
        
        if len(radargrams) == 1:
            axs = [axs]

        ringing_quantifications = []

        for idx, radargram in enumerate(radargrams):
            
            threshold = 8e4

            ringing_amplitudes = [self.quantify_ringing(trace, 10, threshold) for trace in radargram]  # Example values for window_size and threshold
            ringing_quantifications.append(ringing_amplitudes)
            
            im = axs[idx].imshow(radargram.T, aspect='auto', cmap='seismic',
                             extent=[0, radargram.shape[0], radargram.shape[1], 0])
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

        self.radargrams = []  # Initialize radargrams as an empty list
        for file_path in file_paths:
            radargram = self.read_sgy_file(file_path)
            self.radargrams.append(radargram)

        # if self.radargrams:
        #     self.plot_radargrams(self.radargrams)

class Analysis:

    def __init__(self, radargrams):
        self.radargrams = radargrams

    def energy(self, trace):
        return np.sum(trace ** 2)

    def entropy(self, trace):
        # Calculate probability trace, but avoid dividing by zero if sum is zero
        sum_of_squares = np.sum(np.square(trace))
        if sum_of_squares == 0:
            return 0  # If the trace is completely zero, entropy is zero.
        
        p_trace = np.square(trace) / sum_of_squares
        # Calculate entropy, but avoid log of zero by only considering non-zero elements
        entropy = -np.sum(p_trace[p_trace > 0] * np.log(p_trace[p_trace > 0]))
        return entropy

    def dominant_frequency(self, trace, sampling_rate):
        w = fft(trace)
        frequencies = np.fft.fftfreq(len(w), d=1./sampling_rate)
        peak_frequency = frequencies[np.argmax(np.abs(w))]
        return peak_frequency

    def statistical_descriptors(self, trace):
        mean = np.mean(trace)
        median = np.median(trace)
        var = np.var(trace)
        skewness = skew(trace)
        kurt = kurtosis(trace)
        return mean, median, var, skewness, kurt

    def reflectivity(self, trace, window_size):
        windowed_trace = trace[:len(trace) - len(trace) % window_size]
        reshaped_trace = windowed_trace.reshape(-1, window_size)
        rms_amplitude = np.sqrt(np.mean(reshaped_trace**2, axis=1))
        return np.sum(rms_amplitude)

    def coherency(self, radargram, window_size=3):
        """
        Compute the coherency of each trace with its adjacent traces.
        
        Args:
        radargram (np.ndarray): 2D array with traces as columns.
        window_size (int): The size of the sliding window to compute coherency.
        
        Returns:
        np.ndarray: 2D array of coherency values.
        """
        # Number of traces
        num_traces = radargram.shape[1]
        # Initialize coherency matrix with zeros
        coherency_matrix = np.zeros((radargram.shape[0], num_traces))

        # Iterate over each trace, except the first and last which cannot have coherency
        for i in range(1, num_traces - 1):
            # Extract the main trace and its neighbors
            main_trace = radargram[:, i]
            prev_trace = radargram[:, i - 1]
            next_trace = radargram[:, i + 1]

            # Compute coherency using a sliding window
            for j in range(window_size, len(main_trace) - window_size):
                window_main = main_trace[j - window_size:j + window_size]
                window_prev = prev_trace[j - window_size:j + window_size]
                window_next = next_trace[j - window_size:j + window_size]

                # Compute the average coherency with previous and next trace
                coherency_matrix[j, i] = (np.corrcoef(window_main, window_prev)[0, 1] +
                                          np.corrcoef(window_main, window_next)[0, 1]) / 2.0

        return coherency_matrix

    def similarity(self, trace1, trace2):
        return np.correlate(trace1, trace2)

    def SubPlotAnalysis(self, radargrams, energies,
                        entropies,
                        dominant_frequencies,
                        stats,
                        reflectivities,
                        coherency,
                        index=0):
        """
        For each radagrams, plot the energy, entropy, dominant frequency, statistical descriptors, reflectivity, and coherency in a subplot.
        """    

        fig, axs = plt.subplots(6, 1, figsize=(10, 5 * 6))
        axs[0].plot(energies)
        axs[0].set_title(f"Energy of Radargram")
        axs[0].set_xlabel("Traces")
        axs[0].set_ylabel("Energy [kJ]")

        axs[1].plot(entropies)
        axs[1].set_title(f"Entropy of Radargram")
        axs[1].set_xlabel("Traces")
        axs[1].set_ylabel("Entropy [ ]")

        axs[2].plot(dominant_frequencies)
        axs[2].set_title(f"Dominant Frequency of Radargram")
        axs[2].set_xlabel("Traces")
        axs[2].set_ylabel("Frequency [Hz]")

        axs[3].plot(reflectivities)
        axs[3].set_title(f"Reflectivity of Radargram")
        axs[3].set_xlabel("Traces")
        axs[3].set_ylabel("Reflectivity [ ]")

        im = axs[4].imshow(coherency.T, aspect='auto', cmap='seismic')
        axs[4].set_title(f"Coherency of Radargram")
        axs[4].set_xlabel("Traces")
        axs[4].set_ylabel("Time samples [ns]")
        # add colorbar
        # cbar = fig.colorbar(im, ax=axs[4], format='%.1e')
        # cbar.set_label('Coherency')

        # Plot the radargram in the last subplot
        im = axs[5].imshow(radargrams.T, aspect='auto', cmap='seismic')
        axs[5].set_title(f"Radargram")
        axs[5].set_xlabel("Traces")
        axs[5].set_ylabel("Time samples [ns]")
        # add colorbar
        # cbar = fig.colorbar(im, ax=axs[5], format='%.1e')
        # cbar.set_label('Amplitude')

        plt.tight_layout()
        plt.savefig(f'radargram_analysis_{index+1}.png')  # Save each figure with its index number
        plt.close(fig)  # Close the figure after saving to avoid displaying it


    def analyze_traces(self):
        results = []

        # Example: use a fixed sampling rate for FFT, replace with actual if available
        sampling_rate = 1.0

        for index, radargram in enumerate(self.radargrams):
            energies = [self.energy(trace) for trace in radargram]
            entropies = [self.entropy(trace) for trace in radargram]
            dominant_frequencies = [self.dominant_frequency(trace, sampling_rate) for trace in radargram]
            stats = [self.statistical_descriptors(trace) for trace in radargram]
            reflectivities = [self.reflectivity(trace, window_size=10) for trace in radargram]
            coherency = self.coherency(radargram, window_size=5)

            result = {
                'energy': energies,
                'entropy': entropies,
                'dominant_frequency': dominant_frequencies,
                'statistical_descriptors': stats,
                'reflectivity': reflectivities,
                'coherency': coherency
            }

            results.append(result)

            self.SubPlotAnalysis(radargram, energies, entropies, dominant_frequencies, stats, reflectivities, coherency, index)

        return results

if __name__ == "__main__":
    plotter = RadarGramPlotter()
    plotter.run()  # This will read the files and populate the radargrams attribute

    if hasattr(plotter, 'radargrams') and plotter.radargrams:
        analysis = Analysis(plotter.radargrams)
        results = analysis.analyze_traces()

    else:
        print("No radargrams to analyze.")
