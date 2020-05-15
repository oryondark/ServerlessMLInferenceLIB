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
model = NeuralNet('weights_dict.h5')
```

### Performance
1. Theano with Scikit-Learn package could upload to AWS Lambda just only one.
2. It is slow than Pytorch. Because Theano need set up G++ environment. It's not using the g++ accelerator, But i can built-in and checked.
3. No implemented hooking model, It is future study for hook to Gradient.

### Contributor
Hyunjune Kim. - email is '4u_olion@naver.com' , You can call me Jey!

### Bigdata Lab in Kookmin University
