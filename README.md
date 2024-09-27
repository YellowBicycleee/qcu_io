# qcu_io

## 计划提供的功能

### 1. io

将***主机端内存***存储进文件。（单进程部分施工完成，MPI多进程部分2024.10-11月施工）

### 2. 数据格式转换（不需要多进程通信，单进程各自完成）

1. QUDA格式even odd precondition及reverse even odd precondition双向转换
2. su(N) gauge：sunw和quda格式互相转换

### 3. 统一内存管理，将来使得QCU基于QCU_IO
