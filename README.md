# Transformer-Based Weakly Supervised Intracerebral Hemorrhage Segmentation Using Image-Level Labels

## Environment Setup
- Ubuntu 18.04, with Python 3.7 and the following python dependencies.
```
pip install -r requirements.txt
```
## Data Preparation 
<details>
<summary>
RSNA 2019
</summary>
  
- Make your data directory like this below
  ``` bash
  Datasets/
  └── rsna
      ├── ID_225701d63.png
      ├── ...
  ```

  </details>

  <details>
  <summary>
  INSTANCE 2022
  </summary>
  
- Make your data directory like this below
  ``` bash
  Datasets/
  └── instance
      ├── 001_slice_0.h5
      ├── 001_slice_1.h5
      ├── ...
  ```

  </details>

## Usage

### Train
Step 1: Run the run.sh script for training
```
bash run.sh
```
Step 2: Run the evaluate.py for evaluating the single-stage ICH segmentation performance
```
bash run_psa.sh
```
