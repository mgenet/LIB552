#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import numpy
import sympy

import LIB552 as lib


################################################################################


class VectorElement(lib.FiniteElement):
    def __init__(self, finite_element, n_components=None, ordering="component-wise"):
        self.finite_element = finite_element
        self.dim = self.finite_element.dim
        self.n_components = n_components if n_components is not None else self.dim
        self.sym_x = self.finite_element.sym_x
        self.shape = self.finite_element.shape
        self.n_nodes = self.finite_element.n_nodes
        self.sym_nodes = self.finite_element.sym_nodes
        self.n_edges = self.finite_element.n_edges
        self.interpolation = self.finite_element.interpolation
        self.n_points = self.finite_element.n_points
        self.sym_points = self.finite_element.sym_points

        self.n_dofs = self.n_components * self.finite_element.n_dofs
        self.dofs_component     = numpy.empty((self.n_dofs), dtype=numpy.int)
        self.dofs_component_dof = numpy.empty((self.n_dofs), dtype=numpy.int)
        self.ordering = ordering
        if (self.ordering == "component-wise"):
            for k_component in range(self.n_components):
                for k_component_dof in range(self.finite_element.n_dofs):
                    k_dof = k_component*self.finite_element.n_dofs + k_component_dof
                    self.dofs_component[k_dof]     = k_component
                    self.dofs_component_dof[k_dof] = k_component_dof
        elif (self.ordering == "point-wise"):
            for k_component_dof in range(self.finite_element.n_dofs):
                for k_component in range(self.n_components):
                    k_dof = k_component_dof*self.n_components + k_component
                    self.dofs_component[k_dof]     = k_component
                    self.dofs_component_dof[k_dof] = k_component_dof
        # print (self.dofs_component)
        # print (self.dofs_component_dof)
        self.dofs_attachement     = [self.finite_element.dofs_attachement    [self.dofs_component_dof[k_dof]] for k_dof in range(self.n_dofs)]
        self.dofs_attachement_idx = [self.finite_element.dofs_attachement_idx[self.dofs_component_dof[k_dof]] for k_dof in range(self.n_dofs)]

        self.sym_phi = self.scalar_to_vector_sympy_array(self.finite_element.sym_phi)

    def scalar_to_vector_sympy_array(self, scalar_array):
        assert (scalar_array.shape == (self.finite_element.n_dofs,))
        vector_array = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_components)
        for k_dof in range(self.n_dofs):
            k_component     = self.dofs_component[k_dof]
            k_component_dof = self.dofs_component_dof[k_dof]
            vector_array[k_dof, k_component] = scalar_array[k_component_dof,]
        return vector_array

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
        return self.finite_element.get_dof_coords(mesh, k_cell, self.dofs_component_dof[k_cell_dof])

    def init_get_phi_int(self, coeff, n=0):
        """
        Initializes the (efficient) computation of the shape functions element integral.
        This function directly uses the equivalent function within the finite element.
        The coefficient, which is a vector here, is stored and used within the get_phi_int function.
        """
        self.finite_element.init_get_phi_int(coeff=1, n=n)
        self.phi_int_coeff = coeff

    def get_phi_int(self, mesh, k_cell, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        if   (self.ordering == "component-wise"):
            loc_vec[0:self.finite_element.n_dofs] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_vec[0:self.finite_element.n_dofs]
            for k_component in range(self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] *= self.phi_int_coeff[k_component]
        elif (self.ordering == "point-wise"):
            loc_vec[0::self.n_components] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component::self.n_components] = loc_vec[0::self.n_components]
            for k_component in range(0,self.n_components):
                loc_vec[k_component::self.n_components] *= self.phi_int_coeff[k_component]

    def init_get_phi_edge_int(self, coeff, n=0):
        """
        Initializes the (efficient) computation of the shape functions edges integrals.
        This function directly uses the equivalent function within the finite element.
        The coefficient, which is a vector here, is stored and used within the get_phi_int function.
        """
        self.finite_element.init_get_phi_edge_int(coeff=1, n=n)
        self.phi_edge_int_coeff = coeff

    def get_phi_edge_int(self, mesh, k_cell, k_cell_edge, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        if   (self.ordering == "component-wise"):
            loc_vec[0:self.finite_element.n_dofs] = self.finite_element._get_phi_edge_int[k_cell_edge](*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_vec[0:self.finite_element.n_dofs]
            for k_component in range(self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] *= self.phi_edge_int_coeff[k_component]
        elif (self.ordering == "point-wise"):
            loc_vec[0::self.n_components] = self.finite_element._get_phi_edge_int[k_cell_edge](*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component::self.n_components] = loc_vec[0::self.n_components]
            for k_component in range(0,self.n_components):
                loc_vec[k_component::self.n_components] *= self.phi_edge_int_coeff[k_component]

    def init_get_phi_phi_int(self, coeff=1, n=0):
        """
        Initializes the (efficient) computation of the shape functions products element integral.
        This function directly uses the equivalent function within the finite element.
        """
        self.finite_element.init_get_phi_phi_int(coeff=coeff, n=n)

    def get_phi_phi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions products element integral."""
        if   (self.ordering == "component-wise"):
            loc_mat[0:self.finite_element.n_dofs, 0:self.finite_element.n_dofs] = self.finite_element._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_mat[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs, k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_mat[0:self.finite_element.n_dofs, :self.finite_element.n_dofs]
        elif (self.ordering == "point-wise"):
            loc_mat[0::self.n_components,0::self.n_components] = self.finite_element._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_mat[k_component::self.n_components,k_component::self.n_components] = loc_mat[0::self.n_components,0::self.n_components]

    def _init_sym_B(self):
        """Computes the (symbolic) symmetric gradient of the shape functions, and stores them as a (n_dofs x dim x dim) sympy Array."""
        assert (self.dim == 2) and (self.n_components == 2),\
            "Only implemented for 2D vectors on 2D elements. Aborting."
        self.finite_element._init_sym_dphi()
        self.sym_B = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.dim, self.dim)
        if   (self.ordering == "component-wise"):
            self.sym_B[:self.finite_element.n_dofs,0,0] = self.finite_element.sym_dphi[:,0]
            self.sym_B[self.finite_element.n_dofs:,1,1] = self.finite_element.sym_dphi[:,1]
            self.sym_B[:self.finite_element.n_dofs,0,1] = self.finite_element.sym_dphi[:,1]/2
            self.sym_B[self.finite_element.n_dofs:,0,1] = self.finite_element.sym_dphi[:,0]/2
            self.sym_B[:self.finite_element.n_dofs,1,0] = self.finite_element.sym_dphi[:,1]/2
            self.sym_B[self.finite_element.n_dofs:,1,0] = self.finite_element.sym_dphi[:,0]/2
        elif (self.ordering == "point-wise"):
            # MG20201111: This should work, right? Cf. https://github.com/sympy/sympy/issues/20410
            # self.sym_B[0::2,0,0] = self.finite_element.sym_dphi[:,0]
            # self.sym_B[1::2,1,1] = self.finite_element.sym_dphi[:,1]
            # self.sym_B[0::2,0,1] = self.finite_element.sym_dphi[:,1]/2
            # self.sym_B[1::2,0,1] = self.finite_element.sym_dphi[:,0]/2
            # self.sym_B[0::2,1,0] = self.finite_element.sym_dphi[:,1]/2
            # self.sym_B[1::2,1,0] = self.finite_element.sym_dphi[:,0]/2
            for k_dof in range(self.finite_element.n_dofs):
                self.sym_B[2*k_dof  ,0,0] = self.finite_element.sym_dphi[k_dof,0]
                self.sym_B[2*k_dof+1,1,1] = self.finite_element.sym_dphi[k_dof,1]
                self.sym_B[2*k_dof  ,0,1] = self.finite_element.sym_dphi[k_dof,1]/2
                self.sym_B[2*k_dof+1,0,1] = self.finite_element.sym_dphi[k_dof,0]/2
                self.sym_B[2*k_dof  ,1,0] = self.finite_element.sym_dphi[k_dof,1]/2
                self.sym_B[2*k_dof+1,1,0] = self.finite_element.sym_dphi[k_dof,0]/2

    def _init_sym_B_B(self, coeff):
        """Computes the (symbolic) products of shape functions symmetric gradients, and stores them as a (n_dofs x n_dofs) sympy Array."""
        assert (self.sym_B.shape == (self.n_dofs, self.dim, self.dim))
        assert (coeff.shape == (self.dim, self.dim, self.dim, self.dim))
        self.sym_B_B = sympy.tensorcontraction(sympy.tensorcontraction(
            sympy.tensorproduct(
                self.sym_B,
                sympy.tensorcontraction(sympy.tensorcontraction(
                    sympy.tensorproduct(
                        sympy.Array(coeff),
                        sympy.permutedims( # Generalization of transpose for high dimension arrays
                            self.sym_B,
                            (1, 2, 0))),
                    (2, 4)), (2, 3))),
            (1, 3)), (1, 2))
        assert (self.sym_B_B.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_B_B_int(self, n=0):
        """Computes the (symbolic) integrals over the element of the of shape functions symmetric gradients products (stiffness matrix)."""
        self.sym_B_B_int = self.finite_element._integrate_array(array=self.sym_B_B, coeff=1, n=n)

    def init_get_B_B_int(self, coeff, n=0):
        """Initializes the (efficient) computation of the shape functions symmetric gradients products element integral."""
        self._init_sym_B()
        self._init_sym_B_B(coeff)
        self._init_sym_B_B_int(n=n)
        self._get_B_B_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_B_B_int,
            modules="numpy")

    def get_B_B_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions derivatives products element integral."""
        loc_mat[:,:] = self._get_B_B_int(*mesh.get_cell_nodes_coords(k_cell))
