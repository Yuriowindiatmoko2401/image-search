# PetBreedClassifier

PetBreedClassifier takes pet images and predicts their breed using ResNet.

## Usage

#### via Docker image (recommended)

```python
from jina import Flow, Document
import numpy as np
	
f = Flow().add(uses='jinahub+docker://PetBreedClassifier')

doc = Document(tensor=np.ones((224, 224, 3), dtype=np.uint8))

with f:
    f.post(on='/index', inputs=doc, on_done=lambda resp: print(resp.docs[0].tags['prob'], resp.docs[0].tags['label']))

```

#### via source code

```python

from jina import Flow, Document
import numpy as np

doc = Document(tensor=np.ones((224, 224, 3), dtype=np.uint8))

f = Flow().add(uses='jinahub://PetBreedClassifier')

with f:
    f.post(on='/index', inputs=doc, on_done=lambda resp: print(resp.docs[0].tags['prob'], resp.docs[0].tags['label']))

```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
