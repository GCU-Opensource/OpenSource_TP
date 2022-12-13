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

### For mask detection model
```bash
# for training 
python train.py

# for inference
python demo.py

# for accuracy evaluation
python test.py
```

### For webcam face detection
```bash
python webcam_face_dect_image.py

# Press key 'c' for screen shot
# Press key 'q' to quit
```

### For smile detection
```bash
mkdir face

cd smile_detection

# for image
python smile_detector_image.py

# for webcam
python smile_detector_cam.py
```

## Reference
### Smile Detection References
* [https://dontrepeatyourself.org/post/smile-detection-with-python-opencv-and-haar-cascade/](https://dontrepeatyourself.org/post/smile-detection-with-python-opencv-and-haar-cascade/)

* [https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php](https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)

### Mask Detection + Webcam Face Detection References
* [https://heegyukim.medium.com/%EB%A7%88%EC%8A%A4%ED%81%AC-%EC%93%B4-%EC%96%BC%EA%B5%B4-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0-31107905a691](https://heegyukim.medium.com/%EB%A7%88%EC%8A%A4%ED%81%AC-%EC%93%B4-%EC%96%BC%EA%B5%B4-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0-31107905a691)

* [https://m.blog.naver.com/PostView.naver?blogId=teach3450&logNo=221941758674&categoryNo=25&proxyReferer=](https://m.blog.naver.com/PostView.naver?blogId=teach3450&logNo=221941758674&categoryNo=25&proxyReferer=)

* [https://velog.io/@jjaa9292/OpenCVProject-1.-%EB%A7%88%EC%8A%A4%ED%81%AC-%EC%B0%A9%EC%9A%A9%EC%97%AC%EB%B6%80-%ED%99%95%EC%9D%B8-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0](https://velog.io/@jjaa9292/OpenCVProject-1.-%EB%A7%88%EC%8A%A4%ED%81%AC-%EC%B0%A9%EC%9A%A9%EC%97%AC%EB%B6%80-%ED%99%95%EC%9D%B8-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0)

### Slack Messaging
* [https://wooiljeong.github.io/python/slack-bot/](https://wooiljeong.github.io/python/slack-bot/)

