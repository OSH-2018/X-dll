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

  (client稍后再说)

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
  
  ```xml
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

  ```xml
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
  
  ```xml
  <configuration>
    <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
    </property>
  </configuration>
  ```
  
- yarn-site.xml

  ```xml
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
  
**注：由于使用的镜像不同，hadoop的配置文件所在的目录也可能不尽相同，但具体配置应该是大同小异的。**

### 运行hadoop

  进行格式化```hadoop namenode -format```
  然后在```/opt/tools/hadoop/sbin```目录下启动```./start-all.sh```

### 客户机

由于我使用的镜像比较精简，很多命令都没有，所以如果用该镜像创建一个client的话，安装神经网络预测所需要的各种包比较麻烦，所以我直接将宿主机作为client访问HDFS集群。

需要在宿主机上安装JDK和hadoop，其中hadoop的配置方法和master、slave节点的配置方法一样。主要是要把宿主机的IP地址和ssh公钥添加到其他节点，其他节点的IP地址和ssh公钥也要添加到宿主机中。

### 遇到的问题

  一个问题：第一次集群启动成功，第二次就失败了，大概是我不小心改了什么配置。如果始终无法解决的话，就直接在实体机上搭建集群,步骤也差不太多。

错误已经得到解决：一方面是因为docker镜像关闭后，保存的IP地址会消失，尽管我已经保存了对镜像的修改；另一方面是因为由于多次格式化，造成namdenode的namespaceID与datanode的namespaceID不一致，从而导致namenode和datanode的断连，slave节点的datanode不能启动(详情参考这个[博客](https://blog.csdn.net/love666666shen/article/details/74350358))。

## 添加神经网络预测模块

快完成了，等完成再写markdown

## 参考资料

1. [使用Docker搭建hadoop集群](https://blog.csdn.net/qq_33530388/article/details/72811705)
  
