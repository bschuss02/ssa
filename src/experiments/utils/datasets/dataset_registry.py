from experiments.utils.datasets.fluencybank_dataset import create_fluencybank_dataset

fluencybank_dataset = create_fluencybank_dataset(
    "/Users/Benjamin/dev/ssa/data/fluencybank/processed"
)
dataset_registry = {"fluencybank": fluencybank_dataset}
