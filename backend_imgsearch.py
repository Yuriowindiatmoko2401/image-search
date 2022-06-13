from jina import Flow
from docarray import DocumentArray
from ExecutorImgToImgSearch.img2img_search import ImageEncoder, SimpleIndexer
import config


if __name__ == "__main__":
    # index initiate docs array all
    docs_array = DocumentArray.from_files(f"data/*")

    # create Flow
    flow = (
        Flow(
            host=config.HOST,
            protocol=config.PROTOCOL,
            port_expose=config.PORT,
            cors=True,
            uvicorn_kwargs={"loop": "asyncio", "http": "httptools"},
        )
        .add(uses=ImageEncoder, name="Encoder")
        .add(uses=SimpleIndexer, name="Indexer")
    )
    with flow:
        flow.post("/index", inputs=docs_array, show_progress=True)
        flow.block()
