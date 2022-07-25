# timm库的使用
PyTorch图像模型（timm）是一个用于最先进的图像分类的库，
包含image models, optimizers, 
schedulers(调度器), augmentations(增强器)。 

# 安装
``` pip install timn ```

# 1.加载模型
Timm最受欢迎的功能之一是其庞大且不断增长的模型架构集合。其中许多模型包含预训练的权重--要么是在PyTorch中原生训练的，
要么是从Jax和TensorFlow等其他库中移植的--这些模型可以很容易地下载和使用。
我们可以列出并查询可用的模型集合，如下图所示。
``` len(timm.list_models('*')) ```
![image](https://user-images.githubusercontent.com/101920684/179390856-f1587739-4303-4991-868d-98b28e44686a.png)

我们也可以使用`pretrained`参数设置为true使其成为具有预训练权重的模型。
![image](https://user-images.githubusercontent.com/101920684/179390895-4424f4b8-f6f4-44c6-b1dd-41ea1d5213f9.png)

也就是说Timm是一连串模型的集合其中有592个模型有预训练权重

为了简单起见，让我们在这里坚持使用熟悉的、经过测试的ResNet模型系列。
![image](https://user-images.githubusercontent.com/101920684/179391060-2c112748-c3c4-4763-a8d6-7d4e397a24fa.png)
['resnet18',
 'resnet18d',
 'resnet26',
 'resnet26d',
 'resnet26t',
 'resnet32ts',
 'resnet33ts',
 'resnet34',
 'resnet34d',
 'resnet50',
 'resnet50_gn',
 'resnet50d',
 'resnet51q',
 'resnet61q',
 'resnet101',
 'resnet101d',
 'resnet152',
 'resnet152d',
 'resnet200d',
 'resnetblur50',
 'resnetrs50',
 'resnetrs101',
 'resnetrs152',
 'resnetrs200',
 'resnetrs270',
 'resnetrs350',
 'resnetrs420',
 'resnetv2_50',
 'resnetv2_50x1_bit_distilled',
 'resnetv2_50x1_bitm',
 'resnetv2_50x1_bitm_in21k',
 'resnetv2_50x3_bitm',
 'resnetv2_50x3_bitm_in21k',
 'resnetv2_101',
 'resnetv2_101x1_bitm',
 'resnetv2_101x1_bitm_in21k',
 'resnetv2_101x3_bitm',
 'resnetv2_101x3_bitm_in21k',
 'resnetv2_152x2_bit_teacher',
 'resnetv2_152x2_bit_teacher_384',
 'resnetv2_152x2_bitm',
 'resnetv2_152x2_bitm_in21k',
 'resnetv2_152x4_bitm',
 'resnetv2_152x4_bitm_in21k']
 
正如我们所看到的，仍然有很多选择！现在我们来探讨一下如何**从这个列表中创建一个模型。**
``` create_model```
![image](https://user-images.githubusercontent.com/101920684/179391250-f85cc648-8db9-4654-b306-5c0a5415ce71.png)
这样创建出的就是一个普通的PyTorch模型

default_cfg可以访问预训练权重
![image](https://user-images.githubusercontent.com/101920684/179391310-34e1fa8e-3f27-49f8-9fc3-ff0363848f8b.png)

timn集合中的model它们能够处理具有不同通道数的输入图像，这对大多数其他库来说都是一个问题。
直观地说，timm通过对少于3个通道的初始卷积层的权重进行求和，或者智能地将这些权重复制到所需的通道数上。
我们可以通过向.create_model传递in_chans参数来指定我们输入图像的通道数量
![image](https://user-images.githubusercontent.com/101920684/179391650-f32ca449-18d1-49e9-bedb-7581b3b88b9b.png)
使用随机张量来表示单通道图像，我们可以看到，该模型已经处理了图像，并返回了预期的输出形状

值得注意的是，虽然这使我们能够使用一个预训练的模型，但输入与模型所训练的图像有很大不同。
因此，我们不应该期望相同的性能水平，并在新数据集上使用模型之前对其进行微调！

![image](https://user-images.githubusercontent.com/101920684/179391968-bd5886c7-53f6-488b-b599-9b29d7cad536.png)


## 2.加载图片
``` import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension 
test
```

## 3.预测

``` import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
```

## 4.微调
可以通过更改分类器(最后一层)来调整任何预先训练过的模型
```
model = timm.create_model('gluon_resnext101_32x4d', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```




