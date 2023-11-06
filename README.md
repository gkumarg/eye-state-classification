# Eye State Classification from EEG #

Project for MLZoomcamp 2023

## Problem

Electroencephalography, commonly known as EEG, is a technique used to capture the brain's electrical activity. It involves the use of wearable headgear fitted with nonintrusive electrodes that rest on the scalp. This project aims to classify the signals into eye-closed vs eye-open state.

## Dataset

This is a multivariate, sequential, time series dataset that can be used for supervised binary classification task.

This dataset was procured from openml and has an ID-1471.

[OPENML download link](https://www.openml.org/data/download/1587924/phplE7q6h)

Information below is from UCI Repository:

Author: Oliver Roesler

Source: UCI, Baden-Wuerttemberg, Cooperative State University (DHBW), Stuttgart, Germany

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analyzing the video frames. '1' indicates the **eye-closed** and '0' the **eye-open** state. All values are in chronological order with the first measured value at the top of the data.

The features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.

Roesler,Oliver. (2013). EEG Eye State. UCI Machine Learning Repository. https://doi.org/10.24432/C57G7J.

- Sample Data
![Sample Data](assets/sample_data_img.jpg)


## Methodology

Steps 1-6 can be seen in EEG_Eye_State_EDA.ipynb notebook

Step 1: EDA

Figure below shows the original feature FC6 with eyeDetection = 1 when the eyes are closed.
There is some noise in the data as we can see from the spikes.

![Feature - Plot](assets/sample_feature-target_plot.png)

Step 2: Data Cleaning

All features are cleaned using Z-scores with any value exceeding 3 standard deviations removed and filled with a linear interpolation of the data.
Sample feature FC6 after cleaning is shown below:

![Alt text](assets/sample_feature-target_plot_cleaned.png)

Step 3: Baseline comparison of models

Step 4: Model Selection

Step 5: Final Model Training

A final XGBoost model was trained. Getting a very accurate model was not the objective for this project.

Step 6: Model Saving

Step 7: Flask webapp for the model deployment

Step 8: Containerizing the app with Docker

![Docker Run](assets/docker_run.jpg)

Results from running the app:

![Webapp-Results](assets/webapp_results.jpg)