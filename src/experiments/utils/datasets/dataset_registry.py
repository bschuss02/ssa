import logging

from experiments.utils.datasets.fluencybank_dataset import create_fluencybank_dataset

logger = logging.getLogger(__name__)

logger.info("Initializing dataset registry...")

logger.info("Creating FluencyBank dataset...")
fluencybank_dataset = create_fluencybank_dataset(
    "/Users/Benjamin/dev/ssa/data/fluencybank/processed"
)
logger.info(
    f"FluencyBank dataset created successfully with {len(fluencybank_dataset)} samples"
)

dataset_registry = {"fluencybank": fluencybank_dataset}
logger.info(
    f"Dataset registry initialized with {len(dataset_registry)} datasets: {list(dataset_registry.keys())}"
)
