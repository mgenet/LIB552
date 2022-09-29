#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import numpy


################################################################################


class DofManager():
    """
    Structure to manage degrees of freedom.
    Mostly used to distribute degrees of freedom.

    Arguments:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.

    Attributes:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.
        n_dofs (uint): Number of dofs.
        local_to_global (numpy.ndarray of numpy.uint): For each cell and local dof idx, the global dof index (mesh.n_cells x finite_element.n_dofs).
        global_to_local (numpy.ndarray of numpy.uint): For each global dof index, one of the potentially multiple corresponding cell and local dof idx (n_dofs x 2).
        dofs_coords (numpy.ndarray of float): For each global dof index, the coordinates (n_dofs x mesh.n_dim).
    """
    def __init__(self, mesh, finite_element):
        self.mesh = mesh
        self.finite_element = finite_element
        self.n_dofs = None
        self.local_to_global = None
        self.global_to_local = None

    def __repr__(self):
        return "DofManager ("\
              +"n_dofs="         +str(self.n_dofs         )+", "\
              +"local_to_global="+str(self.local_to_global)+")"

    def set_connectivity_from_mesh(self, dim=1):
        """
        Sets the dof connectivity.
        This function can only handle cases where there is exactly one (or dim) dof per node, and no edge/face/cell dofs.
        For vector problems, it is assumed that the dof ordering is point-wise.
        """
        assert (self.finite_element.n_dofs == self.mesh.cell.n_nodes * dim),\
            "Number of dofs per element ("+str(self.finite_element.n_dofs)+") is different from number of nodes per cell ("+str(self.mesh.cell.n_nodes)+") times solution dimension ("+str(dim)+"). Aborting."
        assert (all([(attachement == "node") for attachement in self.finite_element.dofs_attachement])) ,\
            "Not all dofs are attached to nodes. Aborting."
        self.n_dofs = self.mesh.n_nodes * dim
        if   (dim == 1):
            self.local_to_global = self.mesh.cells_nodes
        elif (dim == 2):
            self.local_to_global = numpy.empty((self.mesh.n_cells, self.finite_element.n_dofs), dtype=numpy.uint)
            self.local_to_global[:,0::2] = 2*self.mesh.cells_nodes
            self.local_to_global[:,1::2] = 2*self.mesh.cells_nodes+1

    def set_connectivity_only_node_and_cell_dofs_only_one_dof_per_node(self):
        """
        Sets the dof connectivity.
        This function can handle node as well as cell dofs.
        However, it cannot handle multiple dofs per node.
        """
        self.local_to_global = numpy.empty((self.mesh.n_cells, self.finite_element.n_dofs), dtype=numpy.uint)
        self.n_dofs = 0
        nodes_dofs = -numpy.ones(self.mesh.n_nodes, dtype=numpy.int)
        for k_cell in range(self.mesh.n_cells):
            for k_cell_dof in range(self.finite_element.n_dofs):
                if (self.finite_element.dofs_attachement[k_cell_dof] == "node"):
                    k_cell_node = self.finite_element.dofs_attachement_idx[k_cell_dof]
                    k_node = self.mesh.get_cell_node_index(k_cell, k_cell_node)
                    if (nodes_dofs[k_node] == -1):
                        self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                        nodes_dofs[k_node] = self.local_to_global[k_cell, k_cell_dof]
                    else:
                        self.local_to_global[k_cell, k_cell_dof] = nodes_dofs[k_node]
                elif (self.finite_element.dofs_attachement[k_cell_dof] == "cell"):
                    self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                else:
                    assert (0)

    def set_connectivity_only_node_and_cell_dofs(self):
        """
        Sets the dof connectivity.
        This function can handle node as well as cell dofs.
        It can handle multiple dofs per node. However, in this case it is assumed that the ordering of dofs is the same for each node.
        """
        self.local_to_global = numpy.empty((self.mesh.n_cells, self.finite_element.n_dofs), dtype=numpy.uint)
        self.n_dofs = 0
        nodes_dofs = -numpy.ones((self.mesh.n_nodes, self.finite_element.get_n_dofs_attached_to_each_node()), dtype=numpy.int)
        n_dofs_already_attached_to_cell_nodes = numpy.zeros(self.finite_element.n_nodes, dtype=numpy.uint)
        for k_cell in range(self.mesh.n_cells):
            # print ("k_cell = "+str(k_cell))
            n_dofs_already_attached_to_cell_nodes[:] = 0
            for k_cell_dof in range(self.finite_element.n_dofs):
                # print ("k_cell_dof = "+str(k_cell_dof))
                if (self.finite_element.dofs_attachement[k_cell_dof] == "node"):
                    k_cell_node = self.finite_element.dofs_attachement_idx[k_cell_dof]
                    # print ("k_cell_node = "+str(k_cell_node))
                    k_node = self.mesh.get_cell_node_index(k_cell, k_cell_node)
                    # print ("k_node = "+str(k_node))
                    k_node_dof = n_dofs_already_attached_to_cell_nodes[k_cell_node]
                    # print ("k_node_dof = "+str(k_node_dof))
                    if (nodes_dofs[k_node, k_node_dof] == -1):
                        self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                        nodes_dofs[k_node, k_node_dof] = self.local_to_global[k_cell, k_cell_dof]
                    else:
                        self.local_to_global[k_cell, k_cell_dof] = nodes_dofs[k_node, k_node_dof]
                    n_dofs_already_attached_to_cell_nodes[k_cell_node] += 1
                elif (self.finite_element.dofs_attachement[k_cell_dof] == "cell"):
                    self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                else:
                    assert (0)
            assert ((n_dofs_already_attached_to_cell_nodes == self.finite_element.get_n_dofs_attached_to_each_node()).all())

    def set_connectivity(self):
        """
        Sets the dof connectivity.
        This function can handle node as well as edge and cell dofs.
        It can handle multiple dofs per node as well as multiple nodes per edge.
        """
        self.local_to_global = numpy.empty((self.mesh.n_cells, self.finite_element.n_dofs), dtype=numpy.uint)
        self.n_dofs = 0
        nodes_dofs = numpy.full((self.mesh.n_nodes, self.finite_element.get_n_dofs_attached_to_each_node()), -1, dtype=numpy.int)
        edges_dofs = numpy.full((self.mesh.n_edges, self.finite_element.get_n_dofs_attached_to_each_edge()), -1, dtype=numpy.int)
        n_dofs_already_attached_to_cell_nodes = numpy.zeros(self.finite_element.n_nodes, dtype=numpy.uint)
        n_dofs_already_attached_to_cell_edges = numpy.zeros(self.finite_element.n_edges, dtype=numpy.uint)
        edge_nodes = numpy.empty(2, dtype=numpy.uint)
        for k_cell in range(self.mesh.n_cells):
            # print ("k_cell = "+str(k_cell))
            n_dofs_already_attached_to_cell_nodes[:] = 0
            n_dofs_already_attached_to_cell_edges[:] = 0
            for k_cell_dof in range(self.finite_element.n_dofs):
                # print ("k_cell_dof = "+str(k_cell_dof))
                if (self.finite_element.dofs_attachement[k_cell_dof] == "node"):
                    k_cell_node = self.finite_element.dofs_attachement_idx[k_cell_dof]
                    k_node = self.mesh.get_cell_node_index(k_cell, k_cell_node)
                    k_node_dof = n_dofs_already_attached_to_cell_nodes[k_cell_node]
                    if (nodes_dofs[k_node, k_node_dof] == -1):
                        self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                        nodes_dofs[k_node, k_node_dof] = self.local_to_global[k_cell, k_cell_dof]
                    else:
                        self.local_to_global[k_cell, k_cell_dof] = nodes_dofs[k_node, k_node_dof]
                    n_dofs_already_attached_to_cell_nodes[k_cell_node] += 1
                elif (self.finite_element.dofs_attachement[k_cell_dof] == "edge"):
                    k_cell_edge = self.finite_element.dofs_attachement_idx[k_cell_dof]
                    # print ("k_cell_edge = "+str(k_cell_edge))
                    k_edge = self.mesh.get_cell_edge_index(k_cell, k_cell_edge)
                    # print ("k_edge = "+str(k_edge))
                    edge_nodes[:] = self.mesh.get_edge_nodes_index(k_edge)
                    # print ("edge_nodes = "+str(edge_nodes))
                    if (edge_nodes[0] < edge_nodes[1]):
                        k_edge_dof = n_dofs_already_attached_to_cell_edges[k_cell_edge]
                    else:
                        k_edge_dof = self.finite_element.get_n_dofs_attached_to_each_node()-1 - n_dofs_already_attached_to_cell_edges[k_cell_edge]
                    if (edges_dofs[k_edge, k_edge_dof] == -1):
                        self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                        edges_dofs[k_edge, k_edge_dof] = self.local_to_global[k_cell, k_cell_dof]
                    else:
                        self.local_to_global[k_cell, k_cell_dof] = edges_dofs[k_edge, k_edge_dof]
                    n_dofs_already_attached_to_cell_edges[k_cell_edge] += 1
                elif (self.finite_element.dofs_attachement[k_cell_dof] == "cell"):
                    self.local_to_global[k_cell, k_cell_dof] = self.n_dofs; self.n_dofs += 1
                else:
                    assert (0)
            assert ((n_dofs_already_attached_to_cell_nodes == self.finite_element.get_n_dofs_attached_to_each_node()).all())
            assert ((n_dofs_already_attached_to_cell_edges == self.finite_element.get_n_dofs_attached_to_each_edge()).all())

    def get_cell_dofs_idx(self, k_cell):
        """Returns the idx of the dofs of a given cell."""
        return self.local_to_global[k_cell]

    def get_cell_dof_index(self, k_cell, k_cell_dof):
        """Returns the index of a given dof of given cell."""
        return self.local_to_global[k_cell, k_cell_dof]

    def set_inverse_connectivity(self):
        """Sets the inverse dof connectivity."""
        self.global_to_local = numpy.empty((self.n_dofs, 2), dtype=numpy.uint)
        for k_cell in range(self.mesh.n_cells):
            for k_cell_dof in range(self.finite_element.n_dofs):
                self.global_to_local[self.local_to_global[k_cell, k_cell_dof],:] = [k_cell, k_cell_dof]

    def set_dofs_coords(self):
        """Sets the dofs coordinates."""
        if (self.global_to_local is None): self.set_inverse_connectivity()
        self.dofs_coords = numpy.empty((self.n_dofs, self.mesh.dim), dtype=float)
        self.finite_element.init_get_dofs_coords()
        for k_dof in range(self.n_dofs):
            # print(k_dof)
            k_cell, k_cell_dof = self.global_to_local[k_dof]
            # print(k_cell)
            # print(k_cell_dof)
            self.dofs_coords[k_dof,:] = self.finite_element.get_dof_coords(self.mesh, k_cell, k_cell_dof)
            # print(dofs_coords[k_dof])
