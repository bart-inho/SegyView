# SegyView

SegyView is a Python-based tool for visualizing and analyzing SEG-Y formatted seismic data files. The tool provides functionalities to open and read `.sgy` files, detect and quantify ringing artifacts within seismic traces, perform background noise removal, and generate comparative plots of processed seismic profiles.

## Installation

Before running SegyView, ensure that the following Python libraries are installed:
- `scipy`
- `numpy`
- `matplotlib`
- `tkinter`
- `obspy`
- `tqdm`

These can be installed via `pip` using the following command:

```shell
pip install scipy numpy matplotlib tk obspy tqdm
```

Note: `tkinter` typically comes with Python. If it's not installed, you may need to install it using your system's package manager.

## Usage

To run SegyView, navigate to the directory containing `main.py` and execute the following command in the terminal:

```shell
python main.py
```

A file dialog will appear, allowing you to select one or more `.sgy` files to be processed.

## Features

- **File Selection**: Utilize a GUI dialog to choose SEG-Y files for analysis.
- **Ringing Detection**: Identify and quantify ringing effects in seismic traces.
- **Background Removal**: Subtract the mean of the first N traces from each trace to remove background noise.
- **Amplitude Analysis**: Calculate and plot the mean trace amplitude.
- **Coherency Analysis**: Assess the coherency between adjacent traces (commented out in the current version).
- **Trace Comparison**: Compare the traces with the maximum and minimum mean amplitude.
- **Plot Generation**: Automatically save generated plots as PDF files in a specified directory.

## Output

The results of the analysis are saved as PDF files in the `figures` directory, with each file named according to the index of the profile analyzed.

## Contributing

Contributions to SegyView are welcome. Please feel free to fork the repository, make changes, and submit pull requests.