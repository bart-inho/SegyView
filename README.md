# SegyView

SegyView is a Python script for visualizing and analyzing SEG-Y seismic data files. It provides tools for reading SEG-Y files, plotting radargrams, and conducting various analyses on the data.

## Prerequisites

Before using SegyView, make sure you have the following prerequisites installed:

- Python (version 3.x)
- Required Python libraries: `scipy`, `obspy`, `matplotlib`, `numpy`, and `tkinter`.

You can install these libraries using pip:

```bash
pip install scipy obspy matplotlib numpy
```

## Getting Started

1. Clone or download the SegyView repository to your local machine.

2. Open a terminal or command prompt and navigate to the SegyView directory.

3. Run the script using the following command:

   ```bash
   python main.py
   ```

4. The script will open a file dialog that allows you to select one or more SEG-Y files for analysis.

5. Once you've selected the files, the script will read the data and provide various visualization and analysis options.

## Features

### File Selection

- Use the file dialog to select one or more SEG-Y files for analysis.

### Radargram Plotting

- Visualize the selected SEG-Y data as radargrams.
- Indicate minimum amplitude locations in the radargrams.
- Compare ringing quantifications across different radargrams.

### Trace Analysis

- Compute and visualize various trace-based analyses:
  - Energy: Calculate the energy of each trace.
  - Entropy: Measure the entropy of each trace.
  - Dominant Frequency: Determine the dominant frequency of each trace.
  - Statistical Descriptors: Calculate statistics like mean, median, variance, skewness, and kurtosis for each trace.
  - Reflectivity: Compute reflectivity for each trace.
  - Coherency: Analyze coherency between adjacent traces.

### Subplot Analysis

- Generate a subplot for each radargram, displaying multiple analyses side by side.

### Saving Results

- Save the results of the analysis, including radargram plots, as image files.

## Usage

- Use the file dialog to select SEG-Y files for analysis.
- Explore the various analysis options and visualizations provided by the script.
- Analyze and compare different radargrams based on the computed metrics.
- Save the results and radargram plots for future reference.

## Authors

- Barthélémy Anhorn

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request or open an issue.