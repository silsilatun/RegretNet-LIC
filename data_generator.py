import numpy as np


def generate_item_values_uniform_01(sample_size, num_bidders, num_items):
    data = np.random.rand(
        sample_size, num_bidders, num_items
    ).astype(np.float32)
    return data


def generate_item_values_1x2_asymmetric(sample_size):
    value_for_item_1 = np.random.uniform(4.0, 16.0, sample_size).astype(np.float32)
    value_for_item_2 = np.random.uniform(4.0, 7.0, sample_size).astype(np.float32)
    data = np.expand_dims(
        np.stack(
            (value_for_item_1, value_for_item_2), axis=1
        ),
        axis=1
    )
    return data
