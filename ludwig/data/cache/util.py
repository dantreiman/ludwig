import ludwig
from ludwig.constants import DEFAULTS, ENCODER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PREPROCESSING, TYPE
from ludwig.data.cache.types import CacheableDataset
from ludwig.utils.config_utils import merge_fixed_preprocessing_params
from ludwig.utils.data_utils import hash_dict


def calculate_checksum(original_dataset: CacheableDataset, config: dict):
    features = config.get(INPUT_FEATURES, []) + config.get(OUTPUT_FEATURES, []) + config.get("features", [])
    info = {
        "ludwig_version": ludwig.globals.LUDWIG_VERSION,
        "dataset_checksum": original_dataset.checksum,
        "global_preprocessing": config.get(PREPROCESSING, {}),
        "global_defaults": config.get(DEFAULTS, {}),
        "feature_names": [feature[NAME] for feature in features],
        "feature_types": [feature[TYPE] for feature in features],
        "feature_preprocessing": [
            merge_fixed_preprocessing_params(feature[TYPE], feature.get(PREPROCESSING, {}), feature.get(ENCODER, {}))
            for feature in features
        ],
    }
    return hash_dict(info, max_length=None).decode("ascii")
