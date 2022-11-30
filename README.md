# Mask + Smile detection using OpenCV and PyTorch 

This repository contains a simple implementation of a face mask detector using OpenCV and PyTorch.  

Mask detection is done by our custom trained model using PyTorch.  
The model is trained on the [RMFD](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) dataset.  

We provide a simple Pretrained-PyTorch model that can be used for inference.

## Demo
![figures/NoMask.png](figures/NoMask.png)
![figures/Mask.png](figures/Mask.png)

## Installation
Clone the repository and install the requirements.

Before running the code, you should download the dataset.  
[Download Link (Google drive)](https://drive.google.com/file/d/1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp/view)  

After you downloaded images, you should extract the images to the `FaceMask` folder.  
In the `FaceMask` folder, you should have the following structure:  
```
FaceMask
├── AFDB_face_dataset
│   ├── aidai
│   │   ├── 0_0_aidai_ ... .jpg
│   │   ├── 0_1_aidai_ ... .jpg
│   │   ├── ... 
│   ├── anhu
│   │   ├── 0_0_aidai_ ... .jpg
│   │   ├── 0_1_aidai_ ... .jpg
│   │   ├── 0_2_aidai_ ... .jpg
...
├── AFDB_masked_face_dataset
│   ├── aidai
│   │   ├── 0_0_aidai_ ... .jpg
│   │   ├── 0_1_aidai_ ... .jpg
│   │   ├── ...
│   ├── anhu
│   │   ├── 0_0_aidai_ ... .jpg
```

And you should meet the requirements.  
Install below packages using `pip install` command.  
Or, you can install the packages using `conda install` command.  
```
- [PyTorch] - (https://pytorch.org/get-started/locally/)
- [OpenCV]  - (https://opencv.org/)
- [PIL]     - (https://pillow.readthedocs.io/en/stable/)
```


```bash
git clone
cd mask-detection
pip install -r requirements.txt
```

## Usage

```bash
# for training 
python train.py

# for inference
python demo.py

# for accuracy evaluation
python test.py
```