from jina import Executor, DocumentArray, requests
import torch, torchvision
import numpy as np
import os
import urllib.request
from pathlib import Path


class PetBreedClassifier(Executor):
    def __init__(
        self,
        num_breeds: int = 37,
        pretrained_weights: str = "https://raw.githubusercontent.com/Bashirkazimi/pet-breed-classification/master/files/best_model.pth",
        traversal_paths: str = "@r",
        *args,
        **kwargs,
    ):
        """
        PetBreedClassifier is an image classification executor that takes an image of a pet and classifies its breed.
        It uses the 'resnet18' model from torchvision.models pretrained on Oxford cats and dogs dataset on Kaggle:
        https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset

        :param num_breeds: number of pet breeds (number of classes)
        :param pretrained_weights: path to the pretrained model weights
        :param traversal_paths: Used in the encode method an defines traversal on the
            received `DocumentArray`
        :param default_traversal_paths: the default traversal path that model uses to traverse documents
        """

        super().__init__(*args, **kwargs)
        self.num_breeds = num_breeds
        self.pretrained_weights = pretrained_weights
        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, self.num_breeds)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.download_and_load_model_weights()

        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage("RGB"),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.traversal_paths = traversal_paths
        self.batch_size = 1
        self.classToBreed = {
            0: "Egyptian_Mau",
            1: "Persian",
            2: "Ragdoll",
            3: "Bombay",
            4: "Maine_Coon",
            5: "Siamese",
            6: "Abyssinian",
            7: "Sphynx",
            8: "British_Shorthair",
            9: "Bengal",
            10: "Birman",
            11: "Russian_Blue",
            12: "great_pyrenees",
            13: "havanese",
            14: "wheaten_terrier",
            15: "german_shorthaired",
            16: "samoyed",
            17: "boxer",
            18: "leonberger",
            19: "miniature_pinscher",
            20: "shiba_inu",
            21: "english_setter",
            22: "japanese_chin",
            23: "chihuahua",
            24: "scottish_terrier",
            25: "yorkshire_terrier",
            26: "american_pit_bull_terrier",
            27: "pug",
            28: "keeshond",
            29: "english_cocker_spaniel",
            30: "staffordshire_bull_terrier",
            31: "pomeranian",
            32: "saint_bernard",
            33: "basset_hound",
            34: "newfoundland",
            35: "beagle",
            36: "american_bulldog",
        }

    def download_and_load_model_weights(self):
        cache_dir = Path.home() / ".cache" / "jina-models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name = os.path.basename(self.pretrained_weights)
        model_path = cache_dir / file_name
        if not model_path.exists():
            print(f"=> download {self.pretrained_weights} to {model_path}")
            urllib.request.urlretrieve(self.pretrained_weights, model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    @requests
    def classify(self, docs: DocumentArray, parameters: dict, **kwargs):
        """
        Classify all Documents with pet images (stored in the tensor attribute), store the breeds in the 'label'
        attribute of the Documents, store probabilities in the 'prob' attribute of the Documents' tag

        :param docs: Documents sent to the classifier. The docs must have ``tensor`` of the
            shape ``Height x Width x 3``. By default, the input ``tensor`` must
            be an ``ndarray`` with ``dtype=uint8`` or ``dtype=float32``.
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``traversal_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """

        document_batches_generator = DocumentArray(
            filter(
                lambda x: x.tensor is not None,
                docs[parameters.get("traversal_paths", self.traversal_paths)],
            )
        ).batch(batch_size=parameters.get("batch_size", self.batch_size))

        with torch.inference_mode():
            for batch_docs in document_batches_generator:
                tensors_batch = [self.transform(d.tensor) for d in batch_docs]
                tensors_batch = np.stack(tensors_batch)
                tensors_batch = torch.from_numpy(tensors_batch)
                tensors_batch = tensors_batch.to(self.device)
                outputs = torch.nn.Softmax(dim=1)(self.model(tensors_batch))
                probabilities, classes = torch.max(outputs, 1)
                probabilities = probabilities.cpu().numpy().tolist()
                classes = classes.cpu().numpy().tolist()
                labels = [self.classToBreed[c] for c in classes]
                for doc, label, prob in zip(batch_docs, labels, probabilities):
                    doc.tags["prob"] = prob
                    doc.tags["label"] = label
