import torch
import torch.nn as nn
from typing import Tuple

class InnerNode(nn.Module):
    def __init__(self, selected_feature_index:int):
        super(InnerNode,self).__init__()
        self._selected_feature_index = selected_feature_index

    def forward(self, features:Tuple):
        feature = features[self._selected_feature_index]
        mask = self._dt_function(feature) # N, => N,

        decisions = torch.zeros(*mask.shape, dtype=torch.int64, device=feature.device) - 1

        if self.left_node is not None:
            decisions[mask] = self.left_node([feature[mask] for feature in features])
        if self.right_node is not None:
            decisions[~mask] = self.right_node([feature[mask] for feature in features])
        return decisions

class LeafNode(nn.Module):
    def __init__(self, selected_feature_index:int):
        super(LeafNode,self).__init__()
        self._selected_feature_index = selected_feature_index

    def forward(self, features:Tuple):
        feature = features[self._selected_feature_index]
        mask = self._dt_function(feature) # N, => N,

        decisions = torch.zeros(*mask.shape, dtype=torch.int64, device=feature.device) - 1

        if self.left_node is not None:
            decisions[mask] = self.left_node([feature[mask] for feature in features])
        if self.right_node is not None:
            decisions[~mask] = self.right_node([feature[mask] for feature in features])
        return decisions

class DecisionTree(nn.Module):
    def __init__(self):
        super(DecisionTree,self).__init__()

    def forward(self, input:torch.Tensor):
        pass