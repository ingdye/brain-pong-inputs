# brain_pong_inputs

A Pong game paradigm for fMRI neurofeedback experiments.

## 📖 Overview

`brain_pong_inputs` is an updated version of `brain_pong` (https://github.com/CNIR-SIN/brain-pong-press.git) with additional 'mouse press' and 'mouse wheel' component. The purpose of this additional component is (1) to train participants with delayed response and (2) to allow continous response.

Response with mouse buttons: When the 'left' button is pressed, the bar will go up. When the 'right' button is pressed, the bar will go down. Longer press will shift bar farther. When the button is released, bar stops moving. Bar movement is filtered with HRF kernel, so bar will move slowly and smoothly. Other features are identical to previous version.

Response with mouse wheel: When you scroll the wheel upward, the bar will go up. When you scroll the wheel downward, the bar will go down. Greater scroll will move the bar farther. When you stop scrolling, bar stops moving. Bar movement is filtered with HRF kernel, so bar will move slowly and smoothly. Other features are identical to previous version.



### ⚙️ Configuration

All paradigm parameters are configurable via YAML files (see `configs/sample_config.yaml`). All size and speed parameters use normalized units relative to the game window, ensuring consistent relative scaling across different window sizes.


## 💾 Installation

First, clone the current repository:

```bash
git clone https://github.com/CNIR-SIN/brain-pong-inputs.git
```

Set up the project's `conda` environment and install the package: 

```bash
cd brain-pong-inputs
conda env create -f environment.yml
conda activate brain-pong-inputs
pip install -e .
```

Test the install by printing out the help. The help text should print (see below):
```bash
brain_pong_press --help
```

### Examples

**Basic usage** - Run 1 blocks with 5 trials each using default parameters and 6s delay of bar movement after response.:

```bash
brain_pong_press <output_dir> --blocks 1 --trials 5 --input_method "wheel"
```


## 📁 Output

The game saves:
- **Trial-level data**: Frame-by-frame game state data for each trial (`blockXX_trialXX.tsv`)
- **Block summaries**: Trial performance/difficulty details for each block (`blockXX.tsv`)
- **Log files**: Detailed execution logs for each block (`blockXX.log`)