import tensorflow as tf
from typing import List, Optional, Dict, Generator, NamedTuple, Any, Tuple, Union, Mapping


# From https://www.kaggle.com/code/kentaronakanishi/tf2-0-way-to-accumulate-gradients-in-custom-loop/notebook
def accumulated_gradients(gradients: Optional[List[tf.Tensor]],
                          step_gradients: List[Union[tf.Tensor, tf.IndexedSlices]],
                          num_grad_accumulates: int) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates

    return gradients


# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices
