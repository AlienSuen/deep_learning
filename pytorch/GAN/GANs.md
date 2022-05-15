## 1.what is GANs
A class of ml techniques that consists of two networks playing an **adversarial game** againse each other

一类机器学习技术，由两个网络组成，彼此进行对抗性游戏

![image](https://user-images.githubusercontent.com/101920684/168454446-b1089f9a-9860-4cad-b018-91bdfe4c92b3.png)
![image](https://user-images.githubusercontent.com/101920684/168454470-6161db74-0193-4ac6-be54-abd63ca9eea3.png)
![image](https://user-images.githubusercontent.com/101920684/168454489-ca7e4092-dc71-4d98-81ea-cab979fcaef7.png)
![image](https://user-images.githubusercontent.com/101920684/168454496-23a31ce9-5d92-4beb-a0f9-88dc9a77b002.png)

## 2.why does this work?
in the end the Generator generates dollar bills indistinguishable from real ones and the Discriminator is forced to guess(with probability=1/2)

最后生成器生成与真实钞票无法区分的美元钞票，判别器被迫猜测（概率=1/2）


![image](https://user-images.githubusercontent.com/101920684/168454572-e3fc5f2d-6809-4ac9-9260-e9e4be714f9e.png)

scratch meaning they are both randomly initialized at start and then simultaneously trained

从头开始意味着它们都在开始时随机初始化，然后同时进行训练

## What's the loss function?
![image](https://user-images.githubusercontent.com/101920684/168454601-ae4fdf60-6679-40d9-acf5-4f3788a5d48d.png)
![image](https://user-images.githubusercontent.com/101920684/168454636-8d3685a5-8956-480c-a185-bf3a6d62e61e.png)
![image](https://user-images.githubusercontent.com/101920684/168454653-9e73bb4b-0a2b-4f5e-92a1-60f8b95b83c1.png)

