## 1. Installation(main & run).

### Linux/Unix 
#### (We are run on jetson Orin nano - Ubuntu 22.04)

1. Install [Miniconda](https://github.com/conda-forge/miniforge), [conda-forge](https://conda-forge.org/), Python and `git`.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/MJU-HESTIA/HESTIA_SW.git

  cd HESTIA_SW

  conda create -n hestia python=3.11

  conda activate hestia

  python -m pip install --upgrade pip

  pip install -r requirements.txt

  python main_chat.py

  exit
  ```
---
