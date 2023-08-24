# 基于语言增强的图像新类别发现 (参考代码)

## Python开发环境配置

```bash
# 创建虚拟环境
conda create -n dassl python=3.7
conda activate dassl
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge # torch (version >= 1.7.1)

# 安装 Dassl 工具库
cd Dassl.pytorch-master/
pip install -r requirements.txt
python setup.py develop

cd ..
# 安装其他库
pip install -r requirements.txt

# 完成
```
## 文件处理
解压百度网盘：
将A榜数据集、Dassl.pytorch-master、data复制到git目录。

## 数据

已经生成好了，存放在data文件夹下，如果生成需要安装chatglm的官方api 
https://github.com/THUDM/ChatGLM-6B

python ./data/generate_datasets/data_generate.py

## 训练

``` bash
bash scripts/main.sh 
```
生成分类得分文件 `impreds.json`

## 提交格式

选手需要提交两个文件。一个是.txt格式的文件，包含提交结果对应的模型下载链接，命名为readme.txt. 一个是json格式的预测结果文件（即，上一步中生成的结果 impreds.json），命名为impreds.json。json文件的结构为list类型，list中每个元素是单独一个含有80类预测分值的list类型。json文件格式如下所示。将两个文件打包为submit.zip后提交（创建submit文件夹，将两个文件放到submit中再压缩为submit.zip）。
