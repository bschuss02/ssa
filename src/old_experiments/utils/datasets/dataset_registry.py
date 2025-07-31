import logging
from typing import Any, Dict

from experiments.utils.datasets.fluencybank_dataset import FluencyBankDataset

logger = logging.getLogger(__name__)

logger.info("Initializing dataset registry...")

# Create dataset instances without loading data yet (lazy loading)
logger.info("Creating dataset instances (lazy loading)...")
fluencybank_dataset = FluencyBankDataset(
    dataset_path="/Users/Benjamin/dev/ssa/data/fluencybank/processed"
)
logger.info("✓ FluencyBank dataset instance created (not loaded yet)")

dataset_registry: Dict[str, Any] = {"fluencybank": fluencybank_dataset}
logger.info(
    f"✓ Dataset registry initialized with {len(dataset_registry)} datasets: {list(dataset_registry.keys())}"
)
