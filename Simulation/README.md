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


### benchmark选择特征
- 使用大量小文件
- 文件的使用具有可预测性，即来自某应用或者算法，而不是随机产生的
- 为了使应用普遍性,会使用多种多样的benchmark或者日志记录进行预测
