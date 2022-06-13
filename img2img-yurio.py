
import torch
import torch.nn as nn
from jina import Executor, requests
from docarray import Document, DocumentArray
from torchvision import models


class ImageEncoder(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedder = models.resnet101(pretrained=True) # model si pembuat embed image nantinya
        self._embedder.fx = (
            nn.Identity()
        )  # so that the `output of the model is the embedding vector` and `not the classification logits` (headless pretrained model)

    def _uri_to_torch_tensor(self, doc: Document):
        return (
            doc.load_uri_to_image_tensor()
            .set_image_tensor_shape(shape=(224, 224))
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )
        

    @requests
    @torch.inference_mode()
    def predict(self, docs: DocumentArray, **kwargs):
        docs.apply(lambda d : self._uri_to_torch_tensor(d))  # load image from files and reshape make them torch tensors
        embeds = self._embedder(torch.from_numpy(docs.tensors))  # embed with the resnet101
        docs.embeddings = embeds  # store the embedding in the docs
        del docs[:,'tensor'] # delete the tensors as we only want to have the embedding when indexing

class SimpleIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index = DocumentArray(
            storage='sqlite',
            config={'connection': 'index.db','table_name':'image_to_image'},
        )

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._index.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        docs.match(self._index)


from jina import Flow
from docarray import DocumentArray


image_format = "jpg"
docs_array = DocumentArray.from_files(f"test1/*.{image_format}")


f = (
    Flow(cors=True, port_expose=12345, protocol="http")
    .add(uses=ImageEncoder, name="Encoder")
    .add(uses=SimpleIndexer, name="Indexer")
)


with f:
    f.post("/index", inputs=docs_array.shuffle()[0:10],show_progress=True)


from jina import Client
from docarray import DocumentArray


def print_matches(resp):  # the callback function invoked when task is done
    resp.docs.plot_image_sprites()
    for doc in resp.docs:
        for idx, d in enumerate(doc.matches[:3]):  # print top-3 matches
            print(f'[{idx}]{d.scores["cosine"].value:2f}')

        DocumentArray(doc.matches[:3]).plot_image_sprites()

with f:
    c = Client(protocol="http", port=12345)  # connect to localhost:12345
    c.post("/search", inputs=docs_array[0:2], on_done=print_matches)


with f:
    c = Client(protocol="http", port=12345)  # connect to localhost:12345
    c.post("/search", inputs=docs_array[0:5], on_done=print_matches)




