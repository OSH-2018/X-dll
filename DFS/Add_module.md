# 利用docker搭建HDFS集群

## 搭建环境

- 安装docker

- 拉取镜像

  这一步，我在网上找到了一个配置地比较好的具有hadoop环境的镜像

  ```docker pull registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop```

- 创建容器
  
  创建四个容器，分别用作一个master节点、两个slave节点和一个client
  
  ```docker run -i -t --name Master -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  ```docker run -i -t --name Slave1 -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  ```docker run -i -t --name Slave2 -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  ```docker run -i -t --name Client -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

- 配置Java环境
  
  由于该镜像中已经集成了JDK，所以不需要进行这一步操作，这也是选择这个镜像的好处。

- 配置SSH

  将公钥互相添加到~/.ssh/authorized_keys中
  
  将IP地址互相添加到/etc/hosts中
  
## 配置hadoop
