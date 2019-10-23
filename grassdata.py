import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os

class Tree(object):
    class NodeType(Enum):
        BOX = 0  # box node
        ADJ = 1  # adjacency (adjacent part assembly) node
        SYM = 2  # symmetry (symmetric part grouping) node

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None):
            self.box = box          # box feature vector for a leaf node
            self.sym = sym          # symmetry parameter vector for a symmetry node
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right      # right child
            self.node_type = node_type
            self.label = label

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels):
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        label_list = [l for l in labels[0]]
        box_list.reverse()
        sym_param.reverse()
        label_list.reverse()
        queue = []
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                queue.append(Tree.Node(box=box_list.pop(), node_type=Tree.NodeType.BOX, label=label_list.pop()))
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
            elif ops[0, id] == Tree.NodeType.SYM.value:
                node = queue.pop()
                queue.append(Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM))
        assert len(queue) == 1
        self.root = queue[0]


class GRASSDataset(data.Dataset,):
    def __init__(self, dir, models_num=0, transform=None):
        self.dir = dir
        num_examples = len(os.listdir(os.path.join(dir, 'ops')))
        self.transform = transform
        self.trees = []
        for i in range(models_num):
            boxes = torch.from_numpy(loadmat(os.path.join(dir, 'boxes', '%d.mat' % (i+1)))['box']).t().float()
            ops = torch.from_numpy(loadmat(os.path.join(dir, 'ops', '%d.mat' % (i+1)))['op']).int()
            syms = torch.from_numpy(loadmat(os.path.join(dir, 'syms', '%d.mat' % (i+1)))['sym']).t().float()
            labels = torch.from_numpy(loadmat(os.path.join(dir, 'labels', '%d.mat' % (i+1)))['label']).int()
            tree = Tree(boxes, ops, syms, labels)
            self.trees.append(tree)

    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)