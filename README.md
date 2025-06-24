# ConvLSTM-VideoPredicting

一个基于 ConvLSTM + CBAM 的时序图像预测模型，可用于视频帧预测、动态目标检测或时序掩码生成之类的。

主模型定义在 model.py 文件中，其中 Encoder 为多层级的 ConvLSTM 编码器，Decoder 部分由 CBAM 注意力模块 和 两层转置卷积 组成，用于对时空特征进行解码还原。

A temporal image prediction model based on ConvLSTM + CBAM, which can be used for video frame prediction, dynamic target detection or temporal mask generation.

The main model is defined in the model.py file, where the Encoder is a multi-level ConvLSTM, and the Decoder part consists of a CBAM attention module and two layers of transposed convolution, which are used to decode and restore spatiotemporal features.

## 运行 Run the code

运行forcast文件即可，数据集就是video.mp4的那个视频。

Run the forcast.py file, the dataset is the video named video.mp4.