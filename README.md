# MNIST_two_layer
构建两层神经网络分类器，数据集：MINIST；
不使用pytorch，tensorflow等python package，仅使用numpy。
# 项目运行步骤
首先需要下载MNIST数据集mnist_train.csv和mnist_test.csv放到项目文件夹下，不论训练还是测试都需要做这一步
## 训练模型
在命令行中输入以下命令即可训练模型
```python
python two_layer.py --lr 0.5
                    --hidden [100, 64]
                    --L2 1e-5
                    --epoch 500
                    --batch_size 1000
                    --is_test False
                    --checkpoint 'params.json'
                    --visualize False
```
其中lr代表学习率，默认值0.5。<br/>
hidden代表隐藏层的维度，默认隐藏层维度分别为100、64。<br/>
L2代表L2正则化强度，默认值为1e-5。<br/>
epoch为训练的轮数，默认值为500。<br/>
batch_size为每个batch里含有的样本个数，默认值为1000。<br/>
is_test为是否启动测试模式(需要读取模型参数)。<br/>
checkpoint为模型参数的路径。当is_test为True时表示读取模型参数的路径，当is_test为False时表示保存模型参数的路径。<br/>
visualize为是否可视化网络参数。当is_test为True时，若visualize也为True则会可视化网络参数；如果is_test为False或visualize为False时，均不会可视化网络参数。<br/>

测试模型的代码如下
```python
python two_layer.py --is_test True
                    --checkpoint 'params.json'
                    --visualize True
```                    
## 特别说明
由于模型参数较少，因此可以直接从本项目的github中下载下来，就不放在网盘里了，模型参数文件名为"params.json"

## 实验结果
使用默认值的所有参数值，我们可以得到最终的实验结果<br/>
模型在训练集和测试集上的loss曲线如下图：<br/>
![loss曲线](https://github.com/ccwccwccw/MNIST_two_layer/blob/main/loss.png)<br/>
模型在测试集上的accuracy如下图：<br/>
![acc曲线](https://github.com/ccwccwccw/MNIST_two_layer/blob/main/acc.png)<br/>
最终模型在测试集上的accuracy值为0.978。<br/>
第一层网络参数的可视化结果如下：<br/>
![第一层参数](https://github.com/ccwccwccw/MNIST_two_layer/blob/main/first_layer.png)<br/>
第二层网络参数的可视化结果如下：<br/>
![第二层参数](https://github.com/ccwccwccw/MNIST_two_layer/blob/main/second_layer.png)<br/>
