from typing import Dict, List, Callable
import abc
import queue

import numpy as np

from dream import model
from dream.experiment import model as experimentmodel


captions = [
    "cows in a farm",
    "giraffe being hunted in the savannah",
    "people on a rainy day in the park",
    "little child having a snack",
    "snowboard accident in a tree",
    "a person on a wedding",
    "tourists on an airport",
    "a fat american eating a hot dog",
    "carrot dish",
    "birds out for a fish",
    "a cup of coffee",
    "a bully with a bat",
    "rainy traffic lights",
    "a duck in a fridge",
    "baseball home run",
    "a cute girl with a teddy bear",
    "a person with skis in the Alps",
    "old photo",
    "bike riders racing",
    "zebras on the run",
    "truck driver",
    "brushing teeth",
    "a dirty toilet",
    "a clean toilet",
    "sheep and shepherd dog",
    "milk for santa",
    "train toy",
    "surfer and shark",
    "banana desert",
    "fishing on a boat in the ocean",
    "kite with a child",
    "kite on the beach",
    "elephant in its habitat",
    "old people playing chess",
    "police car",
    "meter on a parking",
    "business cat",
    "cat eating",
    "salad with anchovies",
    "old people",
    "fire fighters watering down a house",
    "wooden table",
    "a lazy person watching tv",
    "a glass of white wine",
    "stop by the road",
    "woman drying her hair",
    "mountains",
    "juicy fruit",
    "mom's good old sandwich",
    "a broken clock on the wall",
    "tennis on clay",
    "traffic lights on a sunny day",
    "big books",
    "swimming in a boat",
    "canoe",
    "bear with fish in the river",
    "polar bear",
    "black bear",
    "broccoli",
    "a green edible plant",
    "old keyboard",
    "computer mouse",
    "kitchen sink with dirty dishes",
    "clean kitchen sink",
    "old horse racer",
    "vase with roses",
    "reader on a bench",
    "laptop and tv",
    "a cooking device",
    "a cold place",
    "starting the tv",
    "a banana  tree",
    "a yellow fruit",
    "baking bread",
    "falling asleep",
    "air battle",
    "broken phone",
    "dog catch a frisbee",
    "italian pizza",
    "child with a fork",
    "french fries",
    "a professor",
    "green sofa",
    "tennis ball",
    "hide from rain",
    "pasta",
    "rider on horse",
    "a pond",
    "luxurious house",
    "elephant drinking water",
    "grapes",
    "hunting dog",
    "cup of rice",
    "chicken wings",
    "sandwich cuts",
    "people with a suitcase",
    "road sign",
    "electronic device",
    "green",
    "halloween costume",
]


class ImageMetadataStore(abc.ABC):
    def atomically(self, cb: Callable[[any], None]) -> None:
        raise NotImplementedError("not yet implemented")

    def new_iter(self, tx: any) -> "ImageMetadataIterator":
        raise NotImplementedError("not yet implemented")


class ImageMetadataIterator(abc.ABC):
    def next(self) -> bool:
        raise NotImplementedError("not yet implemented")

    def curr(self) -> model.Image:
        raise NotImplementedError("not yet implemented")


def find_ideal_relevances(
    im_metadata_store: ImageMetadataStore,
    encode_caption_fn: experimentmodel.EncodeCaptionFn,
) -> Dict[str, List[float]]:
    test_caption_codes = {}

    for test_caption in captions:
        # there should always be exactly one code
        test_caption_codes[test_caption] = encode_caption_fn(test_caption)[0]

    return _find_ideal_relevances_for_caption(im_metadata_store, encode_caption_fn, test_caption_codes)


def _find_ideal_relevances_for_caption(
    im_metadata_store: ImageMetadataStore,
    encode_caption_fn: experimentmodel.EncodeCaptionFn,
    test_caption_codes: Dict[str, np.ndarray],
) -> Dict[str, List[float]]:
    ideal_scores_per_caption = {}

    def _cb(tx: any) -> None:
        nonlocal test_caption_codes
        nonlocal ideal_scores_per_caption

        im_iter = im_metadata_store.new_iter(tx)

        pqs = {}

        for caption in test_caption_codes.keys():
            pqs[caption] = queue.PriorityQueue()

        i = 0

        while im_iter.next():
            im = im_iter.curr()

            im_caption_codes = experimentmodel.im_caption_codes(im, encode_caption_fn)

            for test_caption in test_caption_codes.keys():
                test_caption_code = test_caption_codes[test_caption]

                rel_score = experimentmodel.relevance_score_of_codes(im_caption_codes, test_caption_code)

                pqs[test_caption].put(rel_score)

                if pqs[test_caption].qsize() > experimentmodel.MAX_N:
                    # The priority queue gives priority to smaller numbers.
                    pqs[test_caption].get()

            if i % 100 == 0:
                print("iterated ", i, " images from the database")

                for caption, pq in pqs.items():
                    print(f'current best for caption "{caption}": {pq.queue}')

            i += 1

        for caption, pq in pqs.items():
            ideal_scores_per_caption[caption] = pq.queue

    im_metadata_store.atomically(_cb)
    return ideal_scores_per_caption
