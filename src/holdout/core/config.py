"""Named constants and heuristic thresholds.

Kept in one module so every threshold that influences a decision (a split proposal, a plan
rejection) is visible, documented and testable rather than buried in a branch.
"""

from typing import Final

#: Number of cross-validation folds for every splitter.
N_FOLDS: Final = 5

#: Holdout fraction bounds. Below 10% the holdout score is too noisy to audit validation
#: optimism; above 40% too little data is left to develop on.
TEST_SIZE_DEFAULT: Final = 0.2
TEST_SIZE_MIN: Final = 0.1
TEST_SIZE_MAX: Final = 0.4

#: Accuracy is rejected when the minority class is rarer than this: a constant
#: majority-class predictor would already score at least ``1 - threshold``.
ACCURACY_MIN_MINORITY_FRACTION: Final = 0.2

#: A numeric target with at most this many distinct integer values is treated as classes.
CLASSIFICATION_MAX_CLASSES: Final = 20

#: A string/integer column whose unique ratio reaches this is an identifier, not a feature.
ID_UNIQUE_RATIO: Final = 0.95

#: An identifier column repeats per entity (e.g. customer_id over many rows) when its
#: unique ratio is at most this; such a column suggests a group split.
ENTITY_MAX_UNIQUE_RATIO: Final = 0.5

#: Ratio-based identifier detection needs this many non-missing values; on tiny samples
#: every categorical column looks "mostly unique".
MIN_VALUES_FOR_RATIO_RULES: Final = 50

#: A string column is free text when values are long on average and mostly distinct.
TEXT_MIN_MEAN_LENGTH: Final = 30.0
TEXT_MIN_UNIQUE_RATIO: Final = 0.5

#: A string column is a datetime when this share of a sample of its values parses.
DATETIME_MIN_PARSE_RATE: Final = 0.95
DATETIME_SAMPLE_SIZE: Final = 1000

#: A monotonic numeric column with this unique ratio looks like a time/sequence index.
MONOTONIC_MIN_UNIQUE_RATIO: Final = 0.99

#: Number of example values kept per column in a profile (and their maximum length).
N_EXAMPLE_VALUES: Final = 5
EXAMPLE_MAX_CHARS: Final = 50

#: Cardinality above which the default plan prefers target encoding to one-hot encoding.
ONEHOT_MAX_CARDINALITY: Final = 30

#: A class with fewer rows than this makes stratified 5-fold CV unreliable.
MIN_ROWS_PER_CLASS: Final = 2 * N_FOLDS

#: Threads for estimators that support them; small on purpose (laptop, shared CI runners).
N_JOBS: Final = 2
