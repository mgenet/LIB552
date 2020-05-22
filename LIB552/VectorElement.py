#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import numpy
import sympy
from sympy.integrals.intpoly import polytope_integrate # MG20200501: Why do I need to do that?

import LIB552 as lib


################################################################################


class VectorElement(lib.FiniteElement):
    def __init__(self, finite_element):
        self.finite_element = finite_element
        self.dim = self.finite_element.dim
        self.sym_x = self.finite_element.sym_x
        self.shape = self.finite_element.shape
        self.n_nodes = self.finite_element.n_nodes
        self.sym_nodes = self.finite_element.sym_nodes
        self.n_edges = self.finite_element.n_edges
        self.interpolation = self.finite_element.interpolation
        self.n_points = self.finite_element.n_points
        self.sym_points = self.finite_element.sym_points

        self.n_dofs = self.dim * self.finite_element.n_dofs
        self.dofs_attachement = self.dim * self.finite_element.dofs_attachement
        self.dofs_attachement_idx = self.dim * self.finite_element.dofs_attachement_idx
        self.sym_phi = self.scalar_to_vector_array(self.finite_element.sym_phi)

    def scalar_to_vector_array(self, scalar_array):
        return sympy.transpose(sympy.Array(
            scalar_array.tolist() \
            + [0.]*self.finite_element.n_dofs \
            + [0.]*self.finite_element.n_dofs \
            + scalar_array.tolist(),
            (self.dim, self.n_dofs)))

    def init_get_dofs_coords(self):
        """Initializes the (efficient) computation of dofs coordinates."""
        self.finite_element.init_get_dofs_coords()

    def get_dof_coords(self, mesh, k_cell, k_cell_dof):
        """
        Returns the coordinates of a given dof.

        Args:
            mesh (LIB552.Mesh) The mesh.
            k_cell (uint): The cell index.
            k_cell_dof (uint): The local dof index.
        """
        return self.finite_element.get_dofs_coords(mesh, k_cell)[int(k_cell_dof%self.finite_element.n_dofs)]

    def init_get_phi_int(self, n=0):
        """Initializes the (efficient) computation of the shape functions element integral."""
        self.finite_element.init_get_phi_int(n=n)

    def get_phi_int(self, mesh, k_cell, coeff, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        loc_vec[:self.finite_element.n_dofs] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))
        loc_vec[self.finite_element.n_dofs:] = loc_vec[:self.finite_element.n_dofs]
        loc_vec[:self.finite_element.n_dofs] *= coeff[0]
        loc_vec[self.finite_element.n_dofs:] *= coeff[1]

    def init_get_phi_phi_int(self, n=0):
        """Initializes the (efficient) computation of the shape functions products element integral."""
        self.finite_element.init_get_phi_phi_int(n=n)

    def get_phi_phi_int(self, mesh, k_cell, coeff, loc_mat):
        """(Efficient) computation of shape functions products element integral."""
        loc_mat.fill(0.)
        loc_mat[:self.finite_element.n_dofs, :self.finite_element.n_dofs] = self.finite_element._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))
        loc_mat[self.finite_element.n_dofs:, self.finite_element.n_dofs:] = loc_mat[:self.finite_element.n_dofs, :self.finite_element.n_dofs]
        loc_mat *= coeff

    def init_get_dphi_dphi_int(self, n=0):
        """Initializes the (efficient) computation of the shape functions derivatives products element integral."""
        self.finite_element.init_get_dphi_dphi_int(n=n)

    def get_dphi_dphi_int(self, mesh, k_cell, coeff, loc_mat):
        """(Efficient) computation of shape functions derivatives products element integral."""
        loc_mat.fill(0.)
        loc_mat[:self.finite_element.n_dofs, :self.finite_element.n_dofs] = self.finite_element._get_dphi_dphi_int(*mesh.get_cell_nodes_coords(k_cell))
        loc_mat[self.finite_element.n_dofs:, self.finite_element.n_dofs:] = loc_mat[:self.finite_element.n_dofs, :self.finite_element.n_dofs]
        loc_mat *= coeff
