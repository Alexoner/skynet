import numpy as np
from .autodiff import *



def softmax(node: Node):
    expNode = exp_op(node)
    new_node = expNode / reduce_sum(expNode, axis=1)
    new_node.inputs = [node]
    new_node.name = "softmax({name})".format(name=node.name)

    return new_node
