硬件
- 接线，安全操作

Python
- 图像处理的库
- 仿真
- 调的库

Matlab
- Python， Simulink怎么连接
- HIL
- 仿真
（示教器的内容还需要纠正，坐标系正方向正确了）


深度相机
1. RealSense是怎么做到Alignment的？
内参矩阵K etc.

2. RealSense有一个训好的手部模型，是怎么用
（老师那边有一训好的版是吗）
模型参考MediaPipe, OpenPose
![alt text](image-1.png)

3. RealSense挂墙上，买D415还是D435等，有什么区别，预算

D345：https://blog.csdn.net/weixin_40514381/article/details/142514732

LAB 10的Alignment怎么做到的

4. 图像坐标系：https://ww2.mathworks.cn/help/images/image-coordinate-systems.html



算法
东亚男性 vs 机械臂 上臂和前臂平均长度和比例——相似
![alt text](image.png)

1. 以下两个方案达到效果，是否存在优劣
    1.1. 末端姿态模拟（详：通过获取末端(x,y,z)，通过逆运动学求各个关节角度θ）
    1.2. 向量角度法(详：通过获取(x,y,z)计算出)

![alt text](image-2.png)

产出“论文//手册文档”
- 开始记录，之后再整理
示例
![alt text](image-3.png)

参考“文献”

汇报内容