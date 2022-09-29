#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import math
import numpy

import LIB552 as lib


################################################################################


class Cell:
    """
    Cell structure.
    Stores generic information on cell, e.g., number of nodes.
    Mostly used to compute cell information, e.g., cell "volume".

    Attributes:
        cell_type (str) {Vertex, Line, Triangle, Quadrangle, Tetrahedron, Hexahedron}: The cell type.
        n_nodes (int): The cell node number.
    """
    pass


class Cell_Vertex(Cell):
    cell_type = "Vertex"
    n_nodes = 1
    n_edges = 0

    def get_volume(self, nodes):
        assert (len(nodes) == self.n_nodes)
        return 0.


class Cell_Line(Cell):
    cell_type = "Line"
    n_nodes = 2
    n_edges = 0

    def get_volume(self, nodes):
        assert (len(nodes) == self.n_nodes)
        return numpy.linalg.norm(nodes[1]-nodes[0])


class Cell_Triangle(Cell):
    cell_type = "Triangle"
    n_nodes = 3
    n_edges = 3

    def get_volume(self, nodes):
        assert (len(nodes) == self.n_nodes)
        return numpy.linalg.norm(numpy.cross(nodes[1]-nodes[0], nodes[2]-nodes[0]))

    def get_edge_nodes(self, k_edge):
        return [k_edge, (k_edge+1)%self.n_edges]


class Cell_Quadrangle(Cell):
    cell_type = "Quadrangle"
    n_nodes = 4
    n_edges = 4

    def get_volume(self, nodes):
        assert (len(nodes) == self.n_nodes)
        return numpy.linalg.norm(numpy.cross(nodes[2]-nodes[0], nodes[3]-nodes[1]))

    def get_edge_nodes(self, k_edge):
        if   (k_edge == 0):
            return [0, 1]
        elif (k_edge == 1):
            return [2, 3]
        elif (k_edge == 2):
            return [0, 2]
        elif (k_edge == 3):
            return [1, 3]
        else:
            assert (0), "This should not happen. Aborting."


################################################################################


class Mesh:
    """
    Mesh structure.

    Attributes:
        dim (uint) {1,2}: Spatial dimension of the mesh (also of nodes position vector).
        n_nodes (uint): The number of nodes.
        nodes (numpy.ndarray of float): For each node, the coordinates (n_nodes x dim).
        n_cells (uint): The number of cells.
        cell (LIB552.Cell): Cell structure.
        cells_nodes (numpy.ndarray of numpy.uint): For each cell, the nodes idx (n_cells x cell.n_nodes).
        cells_edges (numpy.ndarray of numpy.uint): For each cell, the edges idx (n_cells x cell.n_edges).
        n_edges (uint): The number of edges.
        edge (LIB552.Cell): Edge structure.
        edges_nodes (numpy.ndarray of numpy.uint): For each edges, the nodes idx (n_edges x edge.n_nodes).
    """
    def __init__(self, dim, nodes, cell, cells_nodes):
        assert (dim in (1,2))
        self.dim = dim
        assert (nodes.ndim == 2)
        assert (nodes.shape[1] == self.dim)
        self.nodes = nodes
        self.n_nodes = self.nodes.shape[0]
        self.cell = cell
        assert (cells_nodes.ndim == 2)
        assert (cells_nodes.shape[1] == self.cell.n_nodes)
        self.cells_nodes = cells_nodes
        self.n_cells = self.cells_nodes.shape[0]

        if   (self.dim == 1):
            self.n_edges = 0
            self.edges_nodes = []
            self.cells_edges = []
        elif (self.dim == 2):
            self.n_edges = self.n_nodes + (self.n_cells+1) - 2 # Euler's formula (only works if mesh has no hole, see later)
            # print ("n_edges = "+str(self.n_edges))
            self.edges_nodes = numpy.empty((self.n_edges,                 2), dtype=numpy.uint)
            self.cells_edges = numpy.empty((self.n_cells, self.cell.n_edges), dtype=numpy.uint)
            nodes_edges = [set() for k_node in range(self.n_nodes)]
            cell_nodes = numpy.empty(self.cell.n_nodes, dtype=numpy.uint)
            edge_nodes = numpy.empty(                2, dtype=numpy.uint)
            k_edge = 0
            for k_cell in range(self.n_cells):
                # print ("k_cell = "+str(k_cell))
                cell_nodes[:] = self.cells_nodes[k_cell]
                # print ("cell_nodes = "+str(cell_nodes))
                for k_cell_edge in range(self.cell.n_edges):
                    # print ("k_cell_edge = "+str(k_cell_edge))
                    edge_nodes = cell_nodes[self.cell.get_edge_nodes(k_cell_edge)]
                    # print ("edge_nodes = "+str(edge_nodes))
                    intersection = nodes_edges[edge_nodes[0]] & nodes_edges[edge_nodes[1]]
                    if   (len(intersection) == 1):
                        self.cells_edges[k_cell, k_cell_edge] = intersection.pop()
                    elif (len(intersection) == 0):
                        if (k_edge >= len(self.edges_nodes)): # If mesh has holes, there are actually more edges than predicted by Euler's formula
                            self.n_edges += 1
                            self.edges_nodes = numpy.concatenate((self.edges_nodes, numpy.empty((1,2), dtype=numpy.uint)))
                        self.cells_edges[k_cell, k_cell_edge] = k_edge
                        self.edges_nodes[k_edge] = numpy.sort(edge_nodes)
                        nodes_edges[edge_nodes[0]].add(k_edge)
                        nodes_edges[edge_nodes[1]].add(k_edge)
                        k_edge += 1
                    else: assert (0), "This should not happen. Aborting."
            assert (k_edge == self.n_edges)

    def __repr__(self):
        return "Mesh ("\
              +"dim="        +str(self.dim        )+", "\
              +"n_nodes="    +str(self.n_nodes    )+", "\
              +"nodes="      +str(self.nodes      )+", "\
             +("n_edges="    +str(self.n_edges    )+", ")*(self.dim==2)\
             +("edges_nodes="+str(self.edges_nodes)+", ")*(self.dim==2)\
              +"n_cells="    +str(self.n_cells    )+", "\
              +"cells_nodes="+str(self.cells_nodes)+", "\
             +("cells_edges="+str(self.cells_edges)     )*(self.dim==2)+")"

    def get_cell_nodes_index(self, k_cell):
        """Returns the global index of the nodes of a given cell."""
        return self.cells_nodes[k_cell]

    def get_cell_node_index(self, k_cell, k_cell_node):
        """Returns the global index of a given node of a given cell."""
        return self.cells_nodes[k_cell, k_cell_node]

    def get_cell_edges_index(self, k_cell):
        """Returns the global index of the edges of a given cell."""
        return self.cells_edges[k_cell]

    def get_cell_edge_index(self, k_cell, k_cell_edge):
        """Returns the global index of a given edge of a given cell."""
        return self.cells_edges[k_cell, k_cell_edge]

    def get_edge_nodes_index(self, k_edge):
        """Returns the global index of the nodes of a given edge."""
        return self.edges_nodes[k_edge]

    def get_edge_node_index(self, k_edge, k_edge_node):
        """Returns the global index of a given node of a given edge."""
        return self.edges_nodes[k_edge, k_edge_node]

    def get_cell_nodes_coords(self, k_cell):
        """Returns the coordinates of the nodes of a given cell."""
        return self.nodes[self.cells_nodes[k_cell]]

    def get_cell_node_coords(self, k_cell, k_cell_node):
        """Returns the coordinates of a given node of a given cell."""
        return self.nodes[self.cells_nodes[k_cell, k_cell_node]]

    def get_cell_volume(self, k_cell):
        """Returns the "volume" of a given cell."""
        return self.cell.get_volume(self.get_cell_nodes(k_cell))

    def get_volume(self):
        """Returns the "volume" of the mesh."""
        return numpy.sum([self.get_cell_volume(k_cell) for k_cell in range(self.n_cells)])


################################################################################


def create_interval_mesh(x0, x1, n_cells=1):
    """
    Creates an interval mesh.

    Args:
        x0 (float): The first point of the mesh.
        x1 (float): The last point of the mesh.
        n_cells (uint): The number of cells.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    dim = 1
    n_nodes = n_cells+1
    nodes = numpy.linspace(x0, x1, n_nodes).reshape((n_nodes,dim))
    cell = lib.Cell_Line()
    cells_nodes = numpy.empty((n_cells, cell.n_nodes), dtype=numpy.uint)
    for k_cell in range(n_cells):
        cells_nodes[k_cell,0] = k_cell
        cells_nodes[k_cell,1] = k_cell+1
    return lib.Mesh(dim=dim, nodes=nodes, cell=cell, cells_nodes=cells_nodes)


def create_unit_interval_mesh(n_cells=1):
    """
    Creates a unit interval mesh.

    Args:
        n_cells (uint): The number of cells.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    return create_interval_mesh(x0=0., x1=1., n_cells=n_cells)


def create_unit_triangle_mesh():
    """
    Creates a unit triangle mesh.
    This mesh has only one element.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    dim = 2
    nodes = numpy.array(
        [[0., 0.],
         [1., 0.],
         [0., 1.]])
    cell = lib.Cell_Triangle()
    cells_nodes = numpy.array(
        [[0, 1, 2]], dtype=numpy.uint)
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)


def create_unit_square_triangular_mesh(
        n_cells_x=1,
        n_cells_y=1):
    """
    Creates a unit square mesh.

    Args:
        n_cells_x (uint): The number of cells in x direction.
        n_cells_y (uint): The number of cells in y direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    dim = 2
    n_nodes_x = n_cells_x + 1
    n_nodes_y = n_cells_y + 1
    n_nodes = n_nodes_x * n_nodes_y
    nodes = numpy.empty(
        (n_nodes, dim))
    for k_y in range(n_nodes_y):
        y = k_y/n_cells_y
        for k_x in range(n_nodes_x):
            x = k_x/n_cells_x
            nodes[k_y*n_nodes_x+k_x] = [x, y]
    # print ("nodes = "+str(nodes))
    cell = lib.Cell_Triangle()
    n_cells = 2 * n_cells_x * n_cells_y
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=numpy.uint)
    for k_y in range(n_cells_y):
        for k_x in range(n_cells_x):
            n1 = k_y * n_nodes_x + k_x
            n2 = n1 + 1
            n3 = n1 + n_nodes_x
            n4 = n3 + 1
            k_cell = 2 * (k_y * n_cells_x + k_x)
            # print("k_cell = "+str(k_cell))
            cells_nodes[k_cell  , :] = [n1, n2, n4]
            cells_nodes[k_cell+1, :] = [n1, n4, n3]
    # print ("cells_nodes = "+str(cells_nodes))
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)


def create_unit_square_quadrangular_mesh(
        n_cells_x=1,
        n_cells_y=1):
    """
    Creates a unit square mesh.

    Args:
        n_cells_x (uint): The number of cells in x direction.
        n_cells_y (uint): The number of cells in y direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    dim = 2
    n_nodes_x = n_cells_x + 1
    n_nodes_y = n_cells_y + 1
    n_nodes = n_nodes_x * n_nodes_y
    nodes = numpy.empty(
        (n_nodes, dim))
    for k_y in range(n_nodes_y):
        y = k_y/n_cells_y
        for k_x in range(n_nodes_x):
            x = k_x/n_cells_x
            k_node = k_y*n_nodes_x+k_x
            nodes[k_node] = [x, y]
    cell = lib.Cell_Quadrangle()
    n_cells = n_cells_x * n_cells_y
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=numpy.uint)
    for k_y in range(n_cells_y):
        for k_x in range(n_cells_x):
            n1 = k_y * n_nodes_x + k_x
            n2 = n1 + 1
            n3 = n1 + n_nodes_x
            n4 = n3 + 1
            k_cell = k_y * n_cells_x + k_x
            # print("k_cell = "+str(k_cell))
            cells_nodes[k_cell, :] = [n1, n2, n3, n4]
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)


def create_quarter_disc_triangular_mesh(
        R=1.,
        n_cells_r=1,
        n_cells_c=1):
    """
    Creates a quarter disc mesh of triangles.

    Args:
        R (float): The disc radius.
        n_cells_r (uint): The number of cells in radial direction.
        n_cells_c (uint): The number of cells in circumferential direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    dim = 2
    n_nodes = 1+n_cells_r*(n_cells_c+1)
    nodes = numpy.empty(
        (n_nodes, dim),
        dtype=float)
    k_node = 0
    nodes[k_node,:] = [0.,0.]
    for k_r in range(1, n_cells_r+1):
        # print("k_r = "+str(k_r))
        r = R * (k_r/n_cells_r)
        # print("r = "+str(r))
        for k_c in range(0, n_cells_c+1):
            # print("k_c = "+str(k_c))
            theta = (math.pi/2) * (k_c/n_cells_c)
            # print("theta = "+str(theta*180/math.pi))
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            k_node = 1 + (k_r-1) * (n_cells_c+1) + k_c
            # print("k_node = "+str(k_node))
            nodes[k_node,:] = [x,y]
    # print(nodes)
    cell = lib.Cell_Triangle()
    n_cells = n_cells_c + (n_cells_r-1) * (2*n_cells_c)
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=numpy.uint)
    for k_c in range(0, n_cells_c):
        # print("k_c = "+str(k_c))
        k_cell = k_c
        # print("k_cell = "+str(k_cell))
        cells_nodes[k_cell, :] = [0, 1+k_c, 1+k_c+1]
    for k_r in range(1, n_cells_r):
        # print("k_r = "+str(k_r))
        for k_c in range(0, n_cells_c):
            # print("k_c = "+str(k_c))
            k_cell = n_cells_c + (k_r-1) * (2*n_cells_c) + 2*k_c
            # print("k_cell = "+str(k_cell))
            n1 = 1 + (k_r-1) * (n_cells_c+1) + k_c
            n2 = n1 + (n_cells_c+1)
            n3 = n2 + 1
            n4 = n1 + 1
            cells_nodes[k_cell  , :] = [n1, n2, n3]
            cells_nodes[k_cell+1, :] = [n1, n3, n4]
    # print(cells_nodes)
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)


def create_quarter_disc_quadrangular_mesh(
        R=1.,
        n_cells_r=1,
        n_cells_c=1):
    """
    Creates a quarter disc mesh of triangles.

    Args:
        R (float): The disc radius.
        n_cells_r (uint): The number of cells in radial direction.
        n_cells_c (uint): The number of cells in circumferential direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    assert (n_cells_c < 1/(2**0.5-1) * n_cells_r), "One must have n_cells_c < 1/(sqrt(2)-1) n_cells_r ≈ 2.5 n_cells_r. Aborting."
    dim = 2
    n_nodes = (n_cells_c+1)**2 + (2*n_cells_c+1)*n_cells_r
    # print("n_nodes = "+str(n_nodes))
    nodes = numpy.empty(
        (n_nodes, dim),
        dtype=float)
    DeltaR = R/(n_cells_c+n_cells_r)
    for k_y in range(n_cells_c+1):
        y = k_y * DeltaR
        for k_x in range(n_cells_c+1):
            x = k_x * DeltaR
            k_node = k_y * (n_cells_c+1) + k_x
            nodes[k_node] = [x, y]
    for k_c in range(2*n_cells_c+1):
        # print("k_c = "+str(k_c))
        if (k_c <= n_cells_c):
            a = k_c * DeltaR
            b = n_cells_c * DeltaR
        else:
            a = n_cells_c * DeltaR
            b = (2*n_cells_c - k_c) * DeltaR
        theta = math.atan2(a, b)
        # print("theta = "+str(t*180/math.pi))
        R0 = (a**2 + b**2)**0.5
        # print("R0 = "+str(R0))
        for k_r in range(n_cells_r):
            # print("k_r = "+str(k_r))
            k_node = (n_cells_c+1)**2 + k_c * n_cells_r + k_r
            # print("k_node = "+str(k_node))
            r = R0 + (R-R0) * (k_r+1)/n_cells_r
            # print("r = "+str(r))
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            nodes[k_node,:] = [x,y]
    # print(nodes)
    cell = lib.Cell_Quadrangle()
    n_cells = n_cells_c**2 + 2*n_cells_c*n_cells_r
    # print("n_cells = "+str(n_cells))
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=numpy.uint)
    for k_y in range(n_cells_c):
        for k_x in range(n_cells_c):
            k_cell = k_y * n_cells_c + k_x
            # print("k_cell = "+str(k_cell))
            n1 = k_y * (n_cells_c+1) + k_x
            n2 = n1 + 1
            n3 = n1 + (n_cells_c+1)
            n4 = n3 + 1
            cells_nodes[k_cell, :] = [n1, n2, n3, n4]
    # print(cells_nodes)
    for k_c in range(n_cells_c):
        # print("k_c = "+str(k_c))
        k_cell = n_cells_c**2 + k_c
        # print("k_cell = "+str(k_cell))
        n1 = n_cells_c + k_c * (n_cells_c+1)
        n2 = (n_cells_c+1)**2 + k_c * n_cells_r
        n3 = n1 + (n_cells_c+1)
        n4 = n2 + n_cells_r
        cells_nodes[k_cell, :] = [n1, n2, n3, n4]
    # print(cells_nodes)
    for k_c in range(n_cells_c):
        # print("k_c = "+str(k_c))
        k_cell = n_cells_c**2 + n_cells_c + k_c
        # print("k_cell = "+str(k_cell))
        n1 = (n_cells_c+1)**2 - 1 - k_c
        n2 = (n_cells_c+1)**2 + n_cells_c*n_cells_r + k_c * n_cells_r
        n3 = n1 - 1
        n4 = n2 + n_cells_r
        cells_nodes[k_cell, :] = [n1, n2, n3, n4]
    # print(cells_nodes)
    for k_c in range(2*n_cells_c):
        # print("k_c = "+str(k_c))
        for k_r in range(n_cells_r-1):
            # print("k_r = "+str(k_r))
            k_cell = n_cells_c**2 + 2*n_cells_c + k_c * (n_cells_r-1) + k_r
            # print("k_cell = "+str(k_cell))
            n1 = (n_cells_c+1)**2 + k_c * n_cells_r + k_r
            n2 = n1 + 1
            n3 = n1 + n_cells_r
            n4 = n3 + 1
            cells_nodes[k_cell, :] = [n1, n2, n3, n4]
    # print(cells_nodes)
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)
