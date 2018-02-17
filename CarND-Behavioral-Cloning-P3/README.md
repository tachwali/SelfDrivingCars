# Behaviorial Cloning Project
Udacity Self Driving Car Nanodegree - Behavioral Cloning

# Prerequisites

To run this project, you need [Miniconda](https://conda.io/miniconda.html) installed(please visit [this link](https://conda.io/docs/install/quick.html) for quick installation instructions.)

Also, a [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip) is needed to create further training data and test the trained network. 

# Installation
To create an environment for this project use the following command on the platform:

```
conda env create -f environment.yml
```


After the environment is created, it needs to be activated with the command:

```
source activate py35
```

*Note: 
environment_aws.yml is the conda environment of aws carnd ami that was used for training* 

# Overview

This repository contains starting files for the Behavioral Cloning Project.

In this project, deep neural networks and convolutional neural networks are used to clone driving behavior. The architecture of the network used in this project is [NVIDIA arcitecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The project is conducted in two stages:
## 1) Training
The network is first trained using training [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided with the project. Further training data could've been generated also using the car [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip).
Training was conducted using aws cloud computing services. EC2 node is created with the following configurations:
* AMI: udacity-carnd - ami-c4c4e3a4
* Hardware: g2.8xlarge
* Environment: check environment_aws.yml
Training such a network on the training [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) took few minutes using g2.8xlarge as compared to almost 48 hours on an 8 core i7 machine with 32GB ram. The key is to leverage the GPU hardware as much as possible by making sure of using tensorflow-gpu and keras-gpu installation.

The training is done by following the steps below:
- access an aws node 
- run 
```
source activate carnd-term1
git clone https://github.com/tachwali/SelfDrivingCars.git
cd SelfDrivingCars/CarND-Behavioral-Cloning-P3/
. run_on_aws.sh
```
At the end of executing the script, three output files are created: 
1) model.h5: a trained model descriptor.
2) model_history.p: stores the training history of keras model
3) aws_environment.yml: stores conda environment used to run the training code

*Note:
model.h5 is sensitive to keras version so make sure to have the same keras version used to save and load this h5 file.*


## 2) Testing
First, we obtain the output files from aws using:
```
cd <path-to-CarND-Behavioral-Cloning-P3>
scp carnd@<node_public_ip_address>:/home/carnd/SelfDrivingCars/CarND-Behavioral-Cloning-P3/model.h5 .
```
Then, using two terminals, run the simulator on one, while run the following on the other terminal:
```
python drive.py model.h5 video
```
On the simulator, select the lowest Screen Resolution and fastest graphic quality. Then select autonomous mode. If the model and drive.py file are correct, the car should start moving autonomously. A common mistake that could happen is a mismatch of input data preprocessing between training and testing.
After finishing one lap, stop the simulator, a folder named 'video' is created. Then run the following to create a video from the images stored in 'video' folder:
```
python video.py video --fps 48
```
As a result, video.mp4 is created.


To meet specifications, the project will require submitting five files: 
* model.py: python script to create and train Keras model.
* drive.py: python script to drive the car. 
* model.h5: a descriptor of a trained Keras model.
* README file : a detailed report about the project
* video.mp4: a video recording of a vehicle driving autonomously using trained Keras model around the track for one full lap)


# The Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Details About Files In This Directory

### `model.py`

### `data_util.py`

### `drive.py`

This file requires  a trained keras model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

The following important changes were made on drive.py:
* Applied the same preprocessing function used for input data:  
```
image_array = ut.preprocess_image(image_array, cv2.COLOR_RGB2YUV) ## simulator returns RGB images
```
* used the following car control settings:
```
controller = SimplePIController(0.05, 0.002)
set_speed = 9
```

* Note:
Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.*


### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).


