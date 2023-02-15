# NeuroSeg2
This is an implementation of NeuroSeg-II on Python 3, Keras, and TensorFlow.

## The repository includes:

Source code of NeuroSeg-II built on FPN and ResNet.  
Training code for Neurofinder.  
Testing code for Neurofinder.  
Testing code for mesoscopic two-photon calcium imaging.  
Code of preprocessing.  

## Using NeuroSeg-II

### **Create a new environment**  

[environment.yaml](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/environment.yaml) supports the normal running of NeuroSeg-II. Before using NeuroSeg-II, ensure that the environment is configured according to this file.  
```
conda env create -f environment.yaml
```
### **Dataset preparation** 

In this folder [Neurofinder](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/Neurofinder), We provide two images for testing.  
* In this folder, [leftImg8bit](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/Neurofinder/test/leftImg8bit) stores the two-photon calcium imaging and [gtFine](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/Neurofinder/test/gtFine) stores the corresponding GT.  
* [generate_dataset.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/Neurofinder/generate_dataset.py) is used to generate [image list](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/Neurofinder/imglists). After adding new images, run this code to generate the list for training and test code can read new images.  

### **Model preparation** 

This folder [models](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/models) is used to store the pretrained model and the test model. The test model can be downloaded from our [huggingface](https://huggingface.co/XZH-James/NeuroSeg2/tree/main).  

### **training or testing** 

After abtaining the dataset and model, running [test.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/test.py) to test the image.  
Running [train.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/train.py) to train the new dataset.

### **The result** 

[logs/evalution](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/logs/evalution/Neurofinder) contains the results of the neurons segmentation of NeuroSeg-II.  
* [plt/difference](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/logs/evalution/Neurofinder/plt/difference) stores the segmented image by NeuroSeg-II.  
* [evaluation_log.csv](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/logs/evalution/Neurofinder/evaluation_log.csv) is the score for this test.  

## Other matters

### **Core code** 
In [neuroseg2](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2) are the core code of NeuroSeg-II.  
* [model.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2/model.py) and [utils.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2/utils.py) are the code of overall structure.  
* [Down.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2/Down.py) is the code of FPN+.  
* [attention.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2/attention.py) is the code of attention mechanism.  
* [visualize.py](https://github.com/XZH-James/NeuroSeg2/blob/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/neuroseg2/visualize.py) is the code for visual segmentation result.  

### **Code of preprocessing** 
In [utilities](https://github.com/XZH-James/NeuroSeg2/tree/main/NeuroSeg%E2%85%A1-main/NeuroSeg%E2%85%A1-main/utilities) are the code for preprocessing.  

## Contact information

If you have any questions about this project, please feel free to contact us. Email address: zhehao_xu@qq.com
