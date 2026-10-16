from src.experiments.cli import bootstrap

# Configure CUDA before any tests import numerical/model modules.
bootstrap()
