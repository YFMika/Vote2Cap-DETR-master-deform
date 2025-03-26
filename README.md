# Vote2Cap-DETR-master-deform
This project builds on "End-to-End 3D Dense Captioning with Vote2Cap-DETR" and "Vote2Cap-DETR++." It uses deformable attention similar to Deformable DETR. However, due to the use of KNN, memory consumption is extremely high. Currently, voxelization is being explored to reduce memory usage.

![f9725addd1f565616369e04b776a232a](https://github.com/user-attachments/assets/7382c100-c33a-4e6f-8036-139138ba7efb)


This project achieved a better mAP@0.5 score (0.546795606) during the pretraining phase. However, unfortunately, its performance during the training phase was not as satisfactory.

You can download our results from the following URL: 

pretrain:[baidudisk](https://pan.baidu.com/s/1AepHqoHvI9VTOQSmyzL4cA?pwd=auew).
train:[baidudisk](https://pan.baidu.com/s/1D87osBba5nqyMWuTPd2pwQ?pwd=zt8k).

Our code is tested with PyTorch 2.0.0, CUDA 11.8 and Python 3.8.20.
For project usage, please refer to the following URL: [github](https://github.com/YFMika/3D-DETR-Caption/blob/main/README.md).
