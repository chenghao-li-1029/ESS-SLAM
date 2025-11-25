# ESS-SLAM

Repository for the paper ["Quantized Self-supervised Local Feature for Real-time Robot Indirect VSLAM"](https://ieeexplore.ieee.org/abstract/document/9444777).

A Self-Supervised-Feature-SLAM System with RGB-D Camera. The Feature-based VSLAM system with self-supervised feature detection is referring to [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2).

Please install the Google Chrome plugin **MathJax Plugin for Github** to display the formulas correctly.

## Prerequisites

### System

Ubuntu16.04+

### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

### CUDA & CUDNN
Cuda 10.* and CUDNN are required for Feature Detection Network Inference.
Tested Under **Cuda 10.2** and **Cudnn 7.6.5**

### Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### OpenCV

We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 3.4.0. Tested with OpenCV 3.4.6**.

The cmake command is attached. Please make sure that libgtk2.0-dev, pkg-config and other prerequisites are installed.
```bash
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=$HOME/Sources/opencv_contrib-$OPENCV_VERSION/modules -D BUILD_TIFF=ON -D OPENCV_ENABLE_NONFREE=ON -DBUILD_PNG=ON -DWITH_CUDA=ON -DBUILD_opencv_cudacodec=OFF ..
```

### Eigen3

Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.
```bash
sudo apt-get install libeigen3-dev
```

### DBoW3 and g2o (Included in Thirdparty folder)

We use modified versions of the [DBoW3](https://github.com/rmsalinas/DBow3) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

### LibTorch
[LibTorch](https://pytorch.org/) is required for Feature Detection Network Inference. 
Download the [cxx11 abi libtorch package](https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.1.zip) and copy the subfolder **libtorch** to `~/Sources`

Tested under **Libtorch 1.5.1**

## Build

### Clone the repository
```bash
git clone https://github.com/Merical/ESS-SLAM.git
```

### Build the project
```bash
cd ESS-SLAM
chmod +x build.sh
./build.sh
```
This will create the executable **rgbd_lchp** in LCHP folder.

## Examples

### Associate the sequence
```bash
python scripts/associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```

### Execute
```bash
cd LCHP
./rgbd_lchp path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_trajectory_dir
```

Example:
```bash
./rgbd_lchp ../Vocabulary/lchp_nms4_n8000.yml TUM1.yaml /path/to/rgbd_dataset_freiburg1_xyz /path/to/associations.txt ./
```

---

## Technical Innovation

### 1. Quantized Self-Supervised Local Feature Learning

**Lightweight Network Architecture**: The feature detection and description network is built in a lightweight manner following the practical guidelines of ShuffleNet V2. This design significantly reduces computational cost while maintaining high feature quality, enabling real-time performance on resource-constrained robotic platforms.

**Self-Supervised Learning**: Unlike traditional hand-crafted features (e.g., ORB) or fully supervised learning approaches, our method employs a self-supervised training strategy that does not require manual annotations. The network learns to detect and describe features directly from unlabeled image data through carefully designed training objectives.

**Quantization for Real-Time Inference**: The network utilizes quantization techniques to compress the model and accelerate inference. By reducing the precision of network parameters and activations, the system achieves faster processing speed and lower memory footprint, which is crucial for real-time robot VSLAM applications.

### 2. Network Architecture

The feature detection and description network consists of 2 basic units, as illustrated in Fig. 1:

- **Inverted Residual Unit**: Splits the input tensor along channels evenly. Applies pointwise (1×1 Conv) and depthwise (3×3 DWConv) convolutional layers instead of normal convolutional layers to decrease FLOPs.

- **Downsampling Unit**: Utilizes depthwise convolutional layers with stride = 2 to replace pooling layers, reducing spatial dimensions while preserving information.

The split channels are concatenated after convolution operations. Finally, channel shuffle before the output provides extra receptive field for learning.

<div align=center><img src="./images/Units.png" alt="Units" width="480" /></div>

<div align=center>Fig. 1  Network Basic Units</div>

### 3. Network Optimization Techniques

To reduce the latency of the network, the following operations are applied:

**1) Separable Convolution**: This operation decomposes a standard convolutional layer into depthwise and pointwise convolutional layers, thus both parameter amount and FLOPs decrease significantly. This is a key component for achieving real-time performance on embedded devices.

**2) Folded Batch Normalization**: Batch normalization makes the training process faster and more stable. Once the training is completed, BN layers can be regarded as a simple linear transformation and merged into the preceding layers like convolution and fully-connected layers to decrease the latency. 

Given the long term mean $\mu$ and standard deviation $\delta$, the batch normalization is folded into the weight $\mathbf{W}$ and bias $\mathbf{B}$ of the preceding convolution layer as:

$$
\mathbf{W}_f=\frac{\gamma \mathbf{W}}{\delta}, \mathbf{B}_f=\beta - \frac{\gamma \mu}{\delta}
$$

where $\mathbf{W}_f$ and $\mathbf{B}_f$ denote the weight and bias of the preceding convolutional layer with the BN layer folded, $\gamma$ and $\beta$ denote the Batch Normalization parameters.

**3) Channel Shuffle**: This operation helps information flow across feature channels in convolutional neural networks. It has been used in the ShuffleNet family. As a parameter-free layer, it provides the network with extra receptive domain without increasing the FLOPs.

### 4. Training Strategy

**Initial Training**: The proposed network is trained by a 4-GTX1080Ti workstation. For initial training with the artificial corner dataset, the shared feature extractor and the keypoint branch are trained by Adam optimizer for 80 epochs. The batch size is set to 128, and the learning rate is set to 10E-4 with half decay every 20 epochs.

**Iterative Joint Training**: For iterative joint training, the cropped COCO and KITTI datasets are applied along with 2D transformation data augmentation. The network is also trained by Adam optimizer for 200 epochs. The batch size is set to 32, and the learning rate is set to 10E-4 with a 10% decay every 40 epochs. The joint training is iterated until convergence or for six rounds at most.

### 5. Key Advantages for Robot VSLAM

- **Real-Time Performance**: Through lightweight architecture, quantization, and efficient network design, the system achieves real-time feature extraction suitable for robotic applications.

- **Robustness**: Self-supervised learned features are more robust to various challenging conditions compared to hand-crafted features, improving tracking reliability.

- **Resource Efficiency**: Low computational cost and memory footprint make the system deployable on resource-constrained robotic platforms.

- **End-to-End Optimization**: The entire feature extraction pipeline can be optimized end-to-end, ensuring better integration with the downstream SLAM tasks.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{li2022quantized,
  author={Li, Shenghao and Liu, Shuang and Zhao, Qunfei and Xia, Qiaoyang},
  journal={IEEE/ASME Transactions on Mechatronics}, 
  title={Quantized Self-Supervised Local Feature for Real-Time Robot Indirect VSLAM}, 
  year={2022},
  volume={27},
  number={3},
  pages={1414-1424},
  keywords={Feature extraction;Feature detection;Robustness;Visualization;Real-time systems;Tensors;Task analysis;Descriptor quantization;indirect VSLAM;local feature;robustness;self-supervised learning},
  doi={10.1109/TMECH.2021.3085326}}
```

## License

This project is released under the GPLv3 license. Please see the LICENSE file for more information.

## Acknowledgements

This work is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). We thank the authors for their excellent work.
