# Optimization of Serverless Platform with Theano

### Current Version
1. Torch to Theano Converter <br>
2. no implementation for Tensorflow and other framework

### Converter Flow architecture
![ConverterFlow](Figures/Converter_flow_arch.png)


### Example
1. Converter
```python
from ml_inference.modeling import *
import numpy as np
import torch
hooking_dummy = torch.Tensor(np.random.rand(3,64,64))
weight_parser(dnn_model, 'torch', hooking_dummy)
```

2.restore model.
```python
from ml_inference.modeling import *
model = NeuralNet('weights.h5')
```

### Summary
1. The light package using *theano* with Scikit-Learn can upload to AWS Lambda.
2. It is slow than Pytorch as Theano need not setup G++ environment.
3. This library can't support Tensorflow or MXNet.

### Contributor
**Hyunjune Kim** - email is '4u_olion@naver.com' , You can call me Jey!<br>
**Kyungyong Lee** - my professor is him, an assistant professor in KOOKMIN University.

### Bigdata Lab in Kookmin University
