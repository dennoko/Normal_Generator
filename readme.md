# Normal Map Generator

This program generates normal maps from mask images. It provides various options for customizing the normal map generation process, including slope profiles, radius, strength, and more.

## Requirements

Ensure you have the following Python libraries installed:

- `numpy`
- `opencv-python`
- `Pillow`

You can install them using pip:

```bash
pip install numpy opencv-python pillow
```

## Usage

Run the program from the command line with the following syntax:

```bash
python main.py <input> <output> [options]
```

### Positional Arguments

- `<input>`: Path to the input mask image (e.g., `input.png`).
- `<output>`: Path to save the generated normal map (e.g., `output.png`).

### Optional Arguments

- `--profile`: Slope profile type. Options:
  - `1`: Linear (default)
  - `2`: Logarithmic
  - `3`: Exponential
- `--radius`: Radius for slope generation (default: `15`).
- `--strength`: Strength of the normal map (default: `1.0`).
- `--type`: Normal map type. Options:
  - `1`: DirectX (Y+) (default)
  - `2`: OpenGL (Y-)
- `--save-intermediates`: Save intermediate processing results (e.g., edges, blurred images).
- `--invert`: Invert the mask image before processing.
- `--disable-blurring`: Disable slope generation and use the mask image directly as the height map.

### Example

To generate a normal map with default settings:

```bash
python main.py input.png output.png
```

To generate a normal map with a logarithmic profile, radius of 20, and OpenGL normal map type:

```bash
python main.py input.png output.png --profile 2 --radius 20 --type 2
```

To save intermediate results and invert the mask:

```bash
python main.py input.png output.png --save-intermediates --invert
```

## Output

The program saves the generated normal map to the specified output path. If `--save-intermediates` is used, intermediate results (e.g., edges, blurred images, height maps) are saved in a `processing` directory located in the same folder as the input image.

## License

This project uses the following libraries, each under its respective license:

- NumPy: [BSD License](https://numpy.org/doc/stable/license.html)
- OpenCV: [Apache License 2.0](https://opencv.org/license/)
- Pillow: [MIT License](https://pillow.readthedocs.io/en/stable/about.html#license)

These licenses are included in the `LICENSE` file of this repository. By using this program, you agree to comply with these licenses.