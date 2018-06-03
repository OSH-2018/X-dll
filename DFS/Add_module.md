## 利用docker搭建HDFS集群

### 搭建环境

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
  
  启动SSH```/etc/init.d/ssh start```
  
  生成秘钥```ssh-keygen -t rsa```

  将公钥互相添加到~/.ssh/authorized_keys中
  
  将IP地址互相添加到/etc/hosts中
  
### 配置hadoop
  
  在Master节点进行配置，然后通过scp命令分发到各节点。总共有四个文件需要配置(在/opt/tools/hadoop/etc/hadoop目录下)。
  
- core-site.xml

  (指定namenode的地址和使用hadoop时产生的文件存放目录)
  
  ```
  <configuration>
    <property>
      <name>fs.defaultFS</name>
      <value>hdfs://Master:9000</value>
    </property>
    <property>
      <name>hadoop.tmp.dir</name>
      <value>/hadoop/data</value>
    </property>
  </configuration>
  ```

- hdfs-site.xml

  (指定保存的副本的数量、namenode的存储位置和datanode的存储位置)

  ```
  <configuration>
    <property>
      <name>dfs.replication</name>
      <value>1</value>
    </property>
    <property>
      <name>dfs.datanode.data.dir</name>
      <value>/hadoop/data</value>
    </property>
    <property>
      <name>dfs.namenode.name.dir</name>
      <value>/hadoop/name</value>
    </property>
  </configuration>
  ```
  
- mapred-site.xml
  
  ```
  <configuration>
    <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
    </property>
  </configuration>
  ```
  
- yarn-site.xml

  ```
  <configuration>
    <property>
      <name>yarn.resourcemanager.address</name>
      <value>Master:8032</value>
    </property>
    <property>
      <name>yarn.resourcemanager.scheduler.address</name>
      <value>Master:8030</value> </property> <property>
      <name>yarn.resourcemanager.resource-tracker.address</name>
      <value>Master:8031</value>
    </property>
    <property>
      <name>yarn.resourcemanager.admin.address</name>
      <value>Master:8033</value>
    </property>
    <property>
      <name>yarn.resourcemanager.webapp.address</name>
      <value>Master:8088</value>
    </property>
    <property>
       <name>yarn.nodemanager.aux-services</name>
       <value>mapreduce_shuffle</value>
    </property>
    <property>
      <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
      <value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
  </configuration>
  ```
  
- 修改slave文件

  将/opt/tools/hadoop/etc/hadoop目录下的slave文件修改为
  
  ```
  Slave1
  Slave2
  ```
  
#### 注：由于使用的镜像不同，hadoop的配置文件所在的目录也可能不尽相同，但具体配置应该是大同小异的。
  
### 运行hadoop

  进行格式化```hadoop namenode -format```
  然后在```/opt/tools/hadoop/sbin```目录下启动```./start-all.sh```
  
### 遇到的问题

  一个问题：第一次集群启动成功，第二次就失败了，大概是我不小心改了什么配置。如果始终无法解决的话，就直接在实体机上搭建集群,步骤也差不太多。
  
## 添加神经网络预测模块

  计划采用shell脚本调用预测模块，以后可能还会有一些其他的乃至其他语言的处理模块，都计划采用shell脚本调用。

  ### 一个简单的示例
  
  关于调用LSTM目录的test模块，要保证调用后全局变量的状态一直存在，就要求调用程序不能退出，为此，先编写一个python脚本调用test模块，并在其中设置一个
  无限循环来接受参数，然后再用shell脚本调用这个python脚本。
  
  __exec.py__
  
  ```
  import test
  while True:
	  command = input()
	  if command == "exit":
		  break
	  else:
	  	a = int(command[11])
		  b = int(command[13])
		  c = int(command[16])
		  d = int(command[18])
		  print(test.count(a,b,[c,d]))
  ```
  
  __test.sh__
  
  ```
  #!/bin/sh
  /usr/bin/python3 exec.py
  ```
  这只是个简单的例子，下面要实现的是如何将HDFS下载的文件名作为参数传递给预测模块，将预测模块的输出作为参数传递给HDFS的下载命令。
  
  这里要说明的是，预测模块的输出并不是一个文件名，而是某种模式，我们要根据这种模式来决定要下载的文件。如何处理这种模式可能还需要其他模块来处理。
  
## 参考资料

1. [使用Docker搭建hadoop集群](https://blog.csdn.net/qq_33530388/article/details/72811705)
  
