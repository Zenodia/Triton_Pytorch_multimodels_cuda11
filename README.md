This repository my minimalistic experiment on Triton serving multiple models ( 2 to be exactly with pytorch) using latest CUDA 11 and TensorRT7.X
It is provided as is and is not maintained regularly, this is not meant as an official repo and user discretion is advised.  
big thanks to Anas Abidin aabidin@nvidia.com , his triton and tensorRT deepdive webniar is brilliant, my experiment is completely based on his work! 

# where to get the data 
download the dataset from kaggle [facial expression](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
unzip the fer2013.csv and put it under NB_images folder or you can get the donwloaded csv file from  [my google drive](https://drive.google.com/file/d/1eowMqIJxDe8bM6EacR6ibBR6VUbSqqWl/view?usp=sharing)

# Setting up the environment

below script will build and run a docker image with a data directory mounted and a choosen GPU( I use GPU 6 )
noted that it is using pytorch:20.06 as base image which requires CUDA 11.0 and driver 450 

```
./startDocker.sh 6 <path_to_data_dir>
```
Once inside this container, please use the following script to enable GPU dashboards
```
./installDashboardInDocker.sh
```
once inside this container , launch jupyter notebook using the below script
```
./start_jupyter_lab.sh
```
then open a browser ( I use firefox) and then type in 0.0.0.0:8888 
 
## open another terminal to launch TRITON server  
A separate container for the server needs to be launched using the script 

```
./start_trtis_server.sh 6 <full_path_to_an_empty_dir>
```


you should see something similar to the below screenshot 




![start_triton_server_successfully](<./pics/start_triton_with_empty_dir.JPG>) 


#### TRITON metrics
To launch Grafana dashboards for monitoring of metrics, please run `docker-compose up` from the [monitoring](./monitoring/) folder and navigate to [localhost:3000/](http://localhost:3000). Additional steps [here](./monitoring/readme.md).


# Notebooks

The three notebooks in this repository walkthrough the example steps for using 
1. build and train the first pytorch CNN model, then deploy it with TensorRT, then serve it with Triton step by step 
1a_triton_serve_own_model.ipynb 
2. pull a pre-trained pytorch model from torch model_zoo and then deploy it with TensorRT also serve it with Triton 
1b_model2_pulled_from_pretrained_modelzoo.ipynb
3. [NB3_lung_segmentation_3d](./NB3_lung_segmentation_3d.ipynb) walks through a simple 3D example with a graphdef backend. 
* For replicating the experiments, additional clients can be launched to test inference with multiple models. For ex. 

open yet another terminal, then recursively copy the 1st model to the empty_dir/ to serve your first model ( under the folder custom_plan/ ) 
```
cp -R model_repo/custom_plan/ empty_dir/custom_plan/
```


![serve_the_1st_model](<./pics/serve_the_first_model.JPG>) 

recursively copy the 2nd  model to the empty_dir/ ( under the folder custom_plan2/) to serve your 2nd model 
you should see something similar to the below in triton server updated interactively the newly populated model

```
cp -R model_repo/custom_plan2/ empty_dir/custom_plan2/
```

you should see something similar to the below in triton server updated interactively the newly populated model


![serve_the_2nd_model](<./pics/serve_the_second_model.JPG>) 


# crediting

This experiment uses code and utilities from the following tools were used as part of this code base and are governed by their respective license agreements. These are in addition to tools distributed within the NGC Docker containers ([Pytorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) / [TRITON](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)).

* [MONAI](https://github.com/Project-MONAI/MONAI/blob/master/LICENSE)
* [Netron](https://github.com/lutzroeder/netron/blob/main/LICENSE)
* [NV Dashboard](https://github.com/rapidsai/jupyterlab-nvdashboard/blob/branch-0.4/LICENSE.txt)
