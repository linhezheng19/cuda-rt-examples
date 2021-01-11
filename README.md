# cuda-rt-examples
A simple cuda and tensorrt example. Example of tensorrt is implement with rt network api, not parsers. 

### CUDA Example
Set your cuda location in `CmakeLists.txt` `line 4`.
All notifications in cuda example are in source files in Chinese.
#### Start
```
$ git clone https://github.com/00hz/cuda-rt-examples.git
$ cd cuda-rt-examples
$ mkdir build && cd build
$ cmake .. && make -j
$ ./example1
$ ./example2
```
#### Note
- You may need to learn some basic knowledge about CUDA program.
- Unified memory(cudaMallocManaged) is used after CUDA6.0.

### TensorRT Network API
Set your TensorRT location in `CmakeLists.txt` `line 5`.
All notifications needed in rt_api example are in source files.
#### Start
```
$ git clone https://github.com/00hz/cuda-rt-examples.git
$ cd cuda-rt-examples
$ python3 scripts/sample_model.py
$ mkdir build && cd build
$ cmake .. && make -j
$ ./example3 -e ../weights/alexnet.bin
```