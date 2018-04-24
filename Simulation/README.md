### 说明
本目录实现的是本地的测试

### 模拟假设
内容来自《Support-Based Prefetching Technique for
Hierarchical Collaborative Caching Algorithm to
Improve the Performance of a Distributed File
System》

- **(i)** File block size is 4 KB
- **(ii)** Average communication delay (ACD) required for transferring 4 KB of data from a remote data node to the local data node is 4 ms
- **(iii)** Time required for transferring time stamp and metadata information is 0.125 ms
- **(iv)** The average time required to access a data block from the local disk storage system is 12 milliseconds
- **(v)** Time required for accessing a block in the main memory is 0.005 ms
- **(vi)** Time required to access the block from the remote memory is 4.01 ms
- **(vii)** Time required to transfer a block from a DN present in the different rack (remote DN) to the client node is 6 ms.
- **(viii)** Time required for cache invalidation is 0.125 ms.


### 数据集选择
- 使用大量小文件
- 文件的使用具有可预测性，即来自某应用或者算法，而不是随机产生的
- 为了使应用普遍性,会使用多种多样的benchmark或者日志记录进行预测

##### 筛选
目前大部分现有的benchmark都是测试io性能，而我们首先需要达到的效果使预测的准确率，在使用神经网络的情况下，而且在现有的硬件水平下，性能必然会显著下降，这方面的提升将交给后续的工作者，我们要做的是，提高准确率，所以了解到的benchmark均不适用。

另外，有少部分关于DFS预测的论文，但是他们使用的数据集几乎都没有go公开，而且是使用某种手段产生的，比如说《Automatic ARIMA Time Series Modeling for Adaptive I/O Prefetching》使用的三种数据集：
- PRISM, a numerical simulation code
- ESCAT, a low temperature plasma modeling code
- Cactus, a numerical relativity code
都是用代码产生的，鉴于此，我们也打算自行产生不同的数据集。由于我们对上面这三个应用不是很了解，所以决定使用机器学习领域的代码产生文件读写log.

##### 数据集1
word2vec
