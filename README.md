# Visual Inertial SLAM

### Directory Structure

```
data
|______{dataset_num}.npz
src
|______camera.py
|______localization.py
|______mapping.py
|______slam.py
|______pr3_utils.py.py

results (generated)
|____ *plots and images

environment.yml
main.py
README.md
```

### Usage

The code is arranged in the above directory structure. All the data should reside in `data` folder.


To run the code, a conda virtual environment needs to be created as shown
```
conda env create -f environment.yml
conda activate proj3
```

Now, once the environment is ready, the code can be run using the following command
```
python main.py
```

This will run the code on dataset `10`. To chagne the dataset change the dataset variable on line 12 in `main.py`.

### Results
Once the code run is complete for a dataset, a folder is generated as `results/` which contains all the plots.
