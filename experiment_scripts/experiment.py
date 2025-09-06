from pathlib import Path

from machine_failure_prediction.pipeline.feature_preprocessing import FeatureProcessor
from machine_failure_prediction.pipeline.data_preprocessing import DataProcessor
from machine_failure_prediction.pipeline.model import ModelWrapper
from machine_failure_prediction.config import Config
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        return "results"