Usage:

    poetry install

    Install CUDA
        pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

    Install CPU
        pip install torch


Project structure:
```
│   .gitignore
│   pyproject.toml
│   README.md
|   requirements.txt - requirements exported from poetry
│
├───data
│   ├───processed
│   └───raw  
│
├───models - trained models
│
├───notebooks
│
│       data_exploration.ipynb
│       plot_comparison.ipynb
│       prepare_data.ipynb
│       profiling.ipynb
│       sweep.ipynb
│       train.ipynb
|
├───onnx - proposed models exported to onnx
│
├───reports
│   ├───figures
│   ├───profile
│   └───tables
│
└───src
    ├───pdi
    │   │   constants.py
    │   │   evaluate.py
    │   │   models.py 
    │   │   train.py
    │   │   visualise.py
    │   │          
    │   └─data
    │       constants.py
    │       preparation.py
    │       types.py
    │       utils.py
    │              
    │
    └───tests
```