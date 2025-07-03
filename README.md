A collection of scripts accompanying the publication Hollman et al. (2025) (DOI
to be added upon publishing). Each script creates one of the figures in the
manuscript, and is named accordingly.

## Setup

### Resource Files
These scripts access resource files which have been created by scripts
available
[here](https://github.com/daraghhollman/messenger-region-classification), and
from the [hermpy](https://github.com/daraghhollman/hermpy) package, which carry
out the training, application, and post processing of the model. These resource
files, a description of them, and a link to script that created them are
included below:

| File name | Description | Link |
| --------- | ----------- | ---- |
| `hollman_2025_crossing_list.csv` | The final list of crossings, including post-processing steps | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/post-processing/2_include_hidden_crossings.py) |
| `messenger_mag` | The full MESSENGER MAG (20 Hz) dataset, formated, and in a pandas dataframe pickled to binary | [link](https://github.com/daraghhollman/hermpy/blob/main/hermpy/mag/mag.py#L529) |
| `model_raw_output.csv` | The raw output produced by the random forest | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/application/get_probabilities.py) |
| `new_crossings.csv` | New individual crossings determined from `model_raw_output.csv`, before post-processing steps | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/application/find_crossings_from_probabilities.py) |
| `new_regions.csv` | Regions corresponding to the above crossings determined from `model_raw_output.csv` | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/application/find_crossings_from_probabilities.py) |
| `models` | A python list of random forest models, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |
| `testing_accuracies` | A python list of accurcay scores for each above model, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |
| `testing_confusion_matrices` | A python list of confusion matrices for each above model, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |
| `models_without_ephemeris` | A python list of random forest models wihtout ephemeris features, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |
| `testing_accuracies_without_ephemeris` | A python list of accurcay scores for each above model, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |
| `testing_confusion_matrices_without_ephemeris` | A python list of confusion matrices for each above model, pickled to binary | [link](https://github.com/daraghhollman/messenger-region-classification/blob/main/modelling/train_model.py) |

These files can be automatically installed using [curl](https://curl.se/) by
running this bash script from the repository base directory (you many need to
add execution permissions to the script: `chmod +x
./scripts/download_resources`):
```shell
./scripts/download_resources
```

If on Windows, this script can be run using the Git BASH terminal (see [git for
windows](https://gitforwindows.org/)).

### Python Environment
These scripts were written using Python 3.12.8 with the following packages:

```
kneed==0.8.5
matplotlib==3.10.3
numpy==2.3.1
pandas==2.3.0
scipy==1.16.0
seaborn==0.13.2
spiceypy==6.0.1
```

along with the custom package [hermpy](https://github.com/daraghhollman/hermpy/).

To avoid package conflicts, we recommend creating a new virtual environment.

A `requirements.txt` is included to install these packages with:
```shell
pip install -r requirements.txt
```

hermpy must be installed manually:
```shell
git clone https://github.com/daraghhollman/hermpy/
cd hermpy/
pip install .
```

The hermpy package has further set up described in the repository [README](https://github.com/daraghhollman/hermpy/blob/main/README.md)

## Reproducing Figures

Each file in `./scripts/` creates a figure included in the manuscript, saved to
`./figures/`. Each file can be run independently, however we include a bash
script to automatically run all scripts. (Note that some scripts involve the
loading of large files into memory, requiring > 16 GiB RAM).

```shell
./scripts/run_all
```
