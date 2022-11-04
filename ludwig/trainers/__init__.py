# register trainers

import ludwig.trainers.directpred_trainer  # noqa: F401
import ludwig.trainers.trainer  # noqa: F401

try:
    import ludwig.trainers.trainer_lightgbm  # noqa: F401
except ImportError:
    pass
