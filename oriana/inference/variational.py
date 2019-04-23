# -*- coding: utf-8 -*-
# variational.py
# author : Antoine Passemiers


class VariationalDistribution:
    """Variational distribution under mean-field approximation.

    Attributes: 
        _partitions (:obj:`np.ndarray`): List of tuples where
            first element is a probabilistic node and second
            element is the variational distribution with
            respect to this partition.
    """

    def __init__(self):
        self._partitions = list()

    def add_partition(self, node_p, node_q):
        partition = (node_p, node_q)
        node_q.name = node_p.name + '-variational'
        if partition not in self._partitions:
            self._partitions.append(partition)
