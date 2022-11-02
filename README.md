<h1>UNNC EEE Final Year Project code by Tianchen Song</h1>

<p>Key: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</p>


![image](https://user-images.githubusercontent.com/70959938/199603800-3e2c72b1-012d-4cb6-bd26-e8f54232e754.png)



<p>Inherited from the Autonomous Delivery Vehicle project, I used three UWB anchors and one UWB tag to achieve indoor localization. Meanwhile, thanks to TensorRT, I accelerated the inference speed of the collision space segmentation model used in the previous project to 3x. Moreover, I built an app using swift to enable the remote control between an iPhone and the vehicle through Bluetooth.</p>

![image](https://user-images.githubusercontent.com/70959938/199602103-34c9fdda-86b7-44b5-bdec-6400f08c07f8.png)

<p>Segmentation Network Structure (R. Fan, H. Wang, P. Cai, and M. Liu, ‘SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection’, p. 17.) </p>
The dataset included 1061 outdoor samples and 512 indoor samples. The dataset was splited into training/validation/test set in the ratio of 6:2:2.

 ![image](https://user-images.githubusercontent.com/70959938/199601606-03dbf007-a38b-4c24-bc79-a329ce23e1c3.png)
![image](https://user-images.githubusercontent.com/70959938/199601631-2580a39b-75b3-425c-ba4c-03420e4942b7.png)
![image](https://user-images.githubusercontent.com/70959938/199601665-f924bb61-0e14-4622-b986-08120e6685db.png)
![image](https://user-images.githubusercontent.com/70959938/199601676-c03edc88-b86b-4bc2-9824-bb5b98a6f818.png)\
The output after 400 epoches training


 
The performance comparison with TensorRT acceleration
 
|                         |     Original    |     FP32      |     FP16     |     INT8     |     Acceleration               |
|-------------------------|-----------------|---------------|--------------|--------------|--------------------------------|
|     18-layers ResNet    |     2.11        |     2.54      |     5.21     |     6.70     |     120.38%/246.92%/317.54%    |
|     34-layers ResNet    |     1.71        |     2.12      |     4.85     |     6.34     |     123.98%/283.63%/370.76%    |
|     Frame rate loss     |     18.96%      |     16.54%    |     6.91%    |     5.37%    |                                |


Please check the report for more details about this project:
 [report.pdf](https://github.com/SimonSongg/FYP/files/9924209/UNNC-FYP-Template-updated.on.March.2021.1.pdf)
