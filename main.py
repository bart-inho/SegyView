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
        plt.show()

    def run(self):
        """Main execution function."""  
        file_paths = self.select_files()

        radargrams = []
        for file_path in file_paths:
            radargram = self.read_sgy_file(file_path)
            radargrams.append(radargram)

        if radargrams:
            self.plot_radargrams(radargrams)


if __name__ == "__main__":
    plotter = RadarGramPlotter()
    plotter.run()