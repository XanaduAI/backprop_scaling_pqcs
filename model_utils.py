import jax
import jax.numpy as jnp
from sklearn.utils import gen_batches
from functools import reduce
import operator

def get_from_dict(dict, key_list):
    """
    get a value from a nested dictionary.
    from https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    :param dict: nested dictionary
    :param key_list: list of keys to be accessed
    :return: the desired value
    """
    return reduce(operator.getitem, key_list, dict)

def set_in_dict(dict, keys, value):
    """
    set a value in a nested dictionary
    :param dict: dictionary
    :param keys: list of keys in nested dictionary
    :param value: value to be set
    :return: nested dictionary with new value
    """
    for key in keys[:-1]:
        dict = dict.setdefault(key, {})
    dict[keys[-1]] = value

def get_nested_keys(d, parent_keys=[]):
    """
    returns the nested keys of a nested dictionary
    :param d: nested dictionary
    :return: list, where each element is a list of nested keys.
    """
    keys_list = []
    for key, value in d.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            keys_list.extend(get_nested_keys(value, current_keys))
        else:
            keys_list.append(current_keys)
    return keys_list

def chunk_vmapped_fn(vmapped_fn, start, max_vmap):
    """
    convert a vmapped function to an equivalent function that evaluates in chunks of size
    max_vmap. The behaviour of chunked_fn should be the same as vmapped_fn, but with a
    lower memory cost.

    the input vmapped_fn should have in_axes = (None, None, ..., 0,0,...,0)

    args:
        :param vmapped function: vmapped function with in_axes = (None, None, ..., 0,0,...,0)
        :param start (int): The index where the first 0 appears in in_axes
        :param max_vmap (int): The max chunk size with which to evaluate the function
        :return chunked_fn: chunked version of the function
    """

    def chunked_fn(*args):
        batch_len = len(args[start])
        batch_slices = list(gen_batches(batch_len, max_vmap))
        res = [vmapped_fn(*args[:start],*[arg[slice] for arg in args[start:]]) for slice in batch_slices]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len/max_vmap%1!=0.0:
            diff = len(res[0])-len(res[-1])
            res[-1] = jnp.pad(res[-1],[(0,diff),*[(0,0)]*(len(res[-1].shape)-1)])
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)
    return chunked_fn

def chunk_grad(grad_fn, max_vmap):
    """
    convert a jax.grad function to an equivalent version that evaluated in chunks of size max_vmap

    grad_fn should be of the form jax.grad(fn(params, X, y), argnums=0), where params is a
    dictionary of paramater jnp.arrays, X, y are jnp.arrays with the same size leading axis, and grad_fn
    is a function that is batch evaluated along this axis (i.e. in_axes = (None,0,0)).

    The return function evaluates the function by splitting the batch evaluation into smaller chunks
    of size max_vmap, and has a lower memory footprint.

    args:
        :param model: gradient function with the functional form jax.grad(loss(params, X,y), argnums=0)
        :param max_vmap (int): the size of the chunks
        :return chunked_grad: chunked version of the function
    """

    def chunked_grad(params, X, y):
        batch_slices = list(gen_batches(len(X), max_vmap))
        grads = [grad_fn(params, X[slice], y[slice]) for slice in batch_slices]
        grad_dict = {}
        for key_list in get_nested_keys(params):
            set_in_dict(grad_dict,
                        key_list,
                        jnp.mean(jnp.array([get_from_dict(grad,key_list) for grad in grads]), axis=0))
        return grad_dict
    return chunked_grad


def chunk_loss(loss_fn, max_vmap):
    """
    convert a loss function of the form loss_fn(params, X,y) to an equivalent version that
    evaluates loss_fn in chunks of size max_vmap. loss_fn should batch evaluate along the leading
    axis of X,y (i.e. in_axes = (None,0,0)).
    :param loss_fn: function of form loss_fn(params, X,y)
    :param max_vmap: maximum chunk size
    :return: chunked_loss: chunked version of the function
    """
    def chunked_loss(params,X,y):
        batch_slices = list(gen_batches(len(X), max_vmap))
        res = jnp.array([loss_fn(params,*[X[slice], y[slice]]) for slice in batch_slices])
        return jnp.mean(res)
    return chunked_loss
