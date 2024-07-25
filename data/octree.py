
from common.plot import Plotter
import numpy as np
import sys

class Node:
    def __init__(self, val: int = 0, children=None):
        self.val = val
        self.children = children or [None] * 8
    
    @property
    def is_leaf(self):
        return all(child is None for child in self.children)
    
    def __repr__(self):
        return self._repr_recursive()

    def _repr_recursive(self, depth=0, max_depth=2):
        indent = "  " * depth
        if self.is_leaf:
            return f"{indent}Node(val={self.val})"
        else:
            if depth >= max_depth:
                return f"{indent}Node(...)"
            children_repr = ",\n".join(self._repr_recursive_child(child, depth + 1, max_depth) for child in self.children)
            return f"{indent}Node(\n{children_repr})"

    def _repr_recursive_child(self, child, depth, max_depth):
        return "  " * depth + "None" if child is None else child._repr_recursive(depth, max_depth)

class Octree:
    def __init__(self, threshold: int = 0):
        self.threshold = threshold
        self.grid_size = 0
        self.root = None
    
    def construct(self, grid: np.ndarray) -> None:
        self.root = self._build_tree(grid, 0, 0, 0, grid.shape[0])
        self.grid_size = grid.shape[0]
    
    def _build_tree(self, grid: np.ndarray, x: int, y: int, z: int, size: int) -> Node:
        if self._is_homogeneous(grid, x, y, z, size):
            return Node(grid[x, y, z])

        half_size = size // 2
        children = [
            self._build_tree(grid, x, y, z, half_size),
            self._build_tree(grid, x, y, z + half_size, half_size),
            self._build_tree(grid, x, y + half_size, z, half_size),
            self._build_tree(grid, x, y + half_size, z + half_size, half_size),
            self._build_tree(grid, x + half_size, y, z, half_size),
            self._build_tree(grid, x + half_size, y, z + half_size, half_size),
            self._build_tree(grid, x + half_size, y + half_size, z, half_size),
            self._build_tree(grid, x + half_size, y + half_size, z + half_size, half_size)
        ]
        
        if all(child.is_leaf for child in children):
            values = {child.val for child in children}
            if len(values) == 1:
                return Node(values.pop())
        
        return Node(children=children)
    
    def _is_homogeneous(self, grid: np.ndarray, x: int, y: int, z: int, size: int) -> bool:
        return np.ptp(grid[x:x+size, y:y+size, z:z+size]) <= self.threshold

    def query(self, x: int, y: int, z: int) -> int:
        return self._query(self.root, 0, 0, 0, self.grid_size, x, y, z)

    def _query(self, node: Node | None, x: int, y: int, z: int, size: int, qx: int, qy: int, qz: int) -> int:
        if node is None:
            raise ValueError("Octree is empty. Maybe you forgot to construct it?")
        
        if node.is_leaf:
            return node.val

        half_size = size // 2
        octant = (
            (qx >= x + half_size) << 2 |
            (qy >= y + half_size) << 1 |
            (qz >= z + half_size)
        )
        
        return self._query(
            node.children[octant],
            x + (octant >> 2) * half_size,
            y + ((octant >> 1) & 1) * half_size,
            z + (octant & 1) * half_size,
            half_size,
            qx, qy, qz
        )
    
    def __len__(self):
        return self._count_nodes(self.root)

    def _count_nodes(self, node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def __sizeof__(self):
        return self._calculate_memory_usage(self.root)
    
    def _calculate_memory_usage(self, node: Node | None) -> int:
        if node is None:
            return 0
        return sys.getsizeof(node) + sum(self._calculate_memory_usage(child) for child in node.children)
    
    def visualize(self, plotter: Plotter, num_slices=8, z_start=30, z_end=120):
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        assert self.root is not None, "Octree is empty. Maybe you forgot to construct it?"
        self._fill_grid(self.root, grid, 0, 0, 0, self.grid_size)
        plotter.plot_irradiance_slices(grid, threshold=self.threshold, num_slices=num_slices, z_start=z_start, z_end=z_end)

    def _fill_grid(self, node: Node, grid: np.ndarray, x: int, y: int, z: int, size: int):
        if node is None or size == 0:
            return
        
        if node.is_leaf:
            grid[x:x+size, y:y+size, z:z+size] = node.val
        else:
            half_size = size // 2
            for i, child in enumerate(node.children):
                if child is not None:
                    self._fill_grid(
                        child,
                        grid,
                        x + ((i >> 2) & 1) * half_size,
                        y + ((i >> 1) & 1) * half_size,  
                        z + (i & 1) * half_size,
                        half_size
                    )