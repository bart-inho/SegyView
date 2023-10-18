# SegyView

SegyView is a tool designed to read and plot radargrams from `.sgy` files. It offers capabilities to visualize the data, "detect" and "quantify" ringing, and compare radargram datasets.

## Features
- Reads `.sgy` files containing radargram data.
- Plots the radargrams with colorbars indicating amplitude.
- Provides functionality to sort of detect and quantify ringing in radargram data.
- Compares ringing quantifications across multiple radargrams.

## Dependencies

- `tkinter`
- `obspy`
- `matplotlib`
- `numpy`

## Installation

Before running the script, ensure you have all the required dependencies. You can install them using `pip`:

```bash
pip install tkinter obspy matplotlib numpy
```

## Usage

1. Run the script:

```bash
python radargram_plotter.py
```

2. A file dialog will open. Select the `.sgy` radargram files you wish to process.
3. The tool will plot each selected radargram, indicating minimum amplitude locations.
4. An additional plot comparing the ringing quantifications across the radargrams will be displayed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
