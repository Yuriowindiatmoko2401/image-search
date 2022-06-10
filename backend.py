from jina import Flow, Document
from ExecutorPetBreedClassifier.pet_breed_classifier import PetBreedClassifier
import config


if __name__ == "__main__":

    # create Flow
    flow = Flow(
        host=config.HOST, protocol=config.PROTOCOL, port_expose=config.PORT
        , cors=True, uvicorn_kwargs={'loop': 'asyncio', 'http': 'httptools'}).add(uses=PetBreedClassifier)
    with flow:
        flow.block()
