# cakewalk

## Purpose

This repository is just for testing

---
## LONGMAN Online Dictionary

**cake‧wake**
```
A very easy thing to do, or a very easy victory. [SYN]Piece of cake.
```

---
## Tensorflow test on Ubuntu 16.04
* **OS warning**

The warning message is like below
```
2017-07-11 00:33:46.655305: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available
on your machine and could speed up CPU computations.
2017-07-11 00:33:46.655333: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available
on your machine and could speed up CPU computations.
2017-07-11 00:33:46.655361: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on
your machine and could speed up CPU computations.
2017-07-11 00:33:46.655366: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on
 your machine and could speed up CPU computations.
2017-07-11 00:33:46.655374: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on
your machine and could speed up CPU computations.
```

To deactive this kind of warning message. Please do the following
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
```
