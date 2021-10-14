# Software Dependencies

1. Python 3.6: <https://www.python.org/downloads/release/python-3615/>
2. TensorFlow 2: <https://www.tensorflow.org/install/>
3. Pillow: <https://pillow.readthedocs.io/en/stable/installation.html>
4. NumPy: <https://www.numpy.org/>
5. Matplotlib: <https://matplotlib.org/install.html>
6. Pandas: <https://pandas.pydata.org/pandas-docs/stable/install.html>
7. Jupyter Notebook: <https://jupyter.org/install.html>

# Setup and Run

1. Download and install all software dependencies listed above.
2. Clone and navigate to our git repository
   [here](https://github.com/Theger14/fyp-ma-13).
3. To run the demonstration code GUI, run the following command:
   
    ```sh
    $ app/app.sh
    ```

4. To view and perform model training and evaluation, run the code in
    `fyp-models/poc_tl_all_cv.ipynb` and `fyp-models/gen_results.py` by
    navigating to `fyp-models` and typing the command:

    ```sh
    $ jupyter notebook
    ```

# Key Components

## `app/app.py`

This is a software script that runs the GUI. It is modelled as a class and
allows the customisation of:

1. The image being selected in either JPG or PNG format
2. The version of the model being used: baseline (unperturbed) or debiased
3. The model architecture: ResNet50, DenseNet, or MobileNet
4. The filter to apply: No filter, glasses filter, makeup filter, and N95 mask
   filter

An image can be loaded with the "Choose an image" button on the left of the
screen and classified with the "Classify" button on the right of the screen.

For detailed documentation for the specific methods and variables used, please
refer to the documentation in `app/app.py`.

## `fyp-models/poc_tl_all_cv.ipynb`

This notebook is focused on model creation, model training, and model
evaluation.

### Outline

1. Import libraries and initialise global variables
2. Load data
3. Data augmentation
4. Load base models
5. Model creation using transfer learning
    - Base models from step 4 are used here
6. Model training
7. Model analysis
    - Get model statistics
8. Findings and results

## `fyp-models/gen_results.py`

This Python script stores useful functions that allow us to perform useful actions, which include:

- Adding a filter to a specific image
- Making a single prediction with a confidence score for a specific image

| Function | Description |
| --- | ----------- |
| ```gen_metrics``` | Generates classification report and confusion matrix (sklearn.metrics). Used by ```gen_save_cr_cm``` function.
| ```gen_save_cr_cm``` | Generates, saves and returns classification reports and confusion matrix. Used in ```poc_tl_all_cv.ipynb```.|
| ```apply_filter``` | Applies a filter to a specific image. Used by the GUI to apply a filter to a specific image and store it in a target file path.|
| ```make_pred``` | Returns predicted class and confidence for a single image. Used by the GUI to get a prediction for a specified image.|
