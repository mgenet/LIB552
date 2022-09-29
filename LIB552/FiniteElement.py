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
from sympy.integrals.intpoly import polytope_integrate # MG20200501: Why do I need to do that?


################################################################################


class FiniteElement():
    """
    Finite element structure.
    Stores all interpolation information.

    Attributes:
        dim (uint) {1,2,3}: The element dimension.

        shape (str) {"Line", "Triangle", "Quadrangle", "Tetrahedron", "Hexahedron"}: The element shape.
        n_nodes (uint): The number of nodes.
        sym_nodes (sympy.Array) (n_nodes x dim): For each node, the (symbolic) coordinates.

        interpolation (str) {"Pk", "Hk"}: The interpolation scheme.
        n_points (uint): The number of points.
        sym_points (sympy.Array) (n_points x dim): For each point, the (symbolic) coordinates.
        n_dofs (uint): The number of degrees of freedom.
        dofs_attachement (list of str): For each dof, the structure to which it is attached ({"node", "edge", "cell"}).
        dofs_attachement_idx (list of uint): For each dof, the local index of the structure to which it is attached.
        sym_phi (sympy.Array) (n_dofs): For each dof, the (symbolic) shape function.
    """
    def __repr__(self):
        return self.__class__.__name__.split("_")[0]+" (" \
              +"dim="          +str(self.dim          )+", " \
              +"shape="        +    self.shape         +", " \
              +"n_nodes="      +str(self.n_nodes      )+", " \
              +"interpolation="+    self.interpolation +", " \
              +"n_points="     +str(self.n_points     )+", " \
              +"n_dofs="       +str(self.n_dofs       )+")"

    def get_n_dofs_attached_to_nodes(self):
        """Returns the number of dofs that are attached to the cell nodes."""
        return self.dofs_attachement.count("node")

    def get_n_dofs_attached_to_each_node(self):
        """Returns the number of dofs that are attached to each cell node."""
        return self.get_n_dofs_attached_to_nodes()//self.n_nodes

    def get_n_dofs_attached_to_edges(self):
        """Returns the number of dofs that are attached to the cell edges."""
        return self.dofs_attachement.count("edge")

    def get_n_dofs_attached_to_each_edge(self):
        """Returns the number of dofs that are attached to each cell edge."""
        if (self.n_edges > 0):
            return self.get_n_dofs_attached_to_edges()//self.n_edges
        else:
            return 0

    def get_n_dofs_attached_to_cell(self):
        """Returns the number of dofs that are attached to the cell."""
        return self.dofs_attachement.count("cell")

    def init_get_dofs_coords(self):
        """Initializes the (efficient) computation of dofs coordinates."""
        self._get_dofs_coords = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_points,
            modules="numpy")

    def get_dofs_coords(self, mesh, k_cell):
        """
        Returns the coordinates of the dofs.

        Args:
            mesh (LIB552.Mesh) The mesh.
            k_cell (uint): The cell index.
        """
        return self._get_dofs_coords(*mesh.get_cell_nodes_coords(k_cell))

    def get_dof_coords(self, mesh, k_cell, k_cell_dof):
        """
        Returns the coordinates of a given dof.

        Args:
            mesh (LIB552.Mesh) The mesh.
            k_cell (uint): The cell index.
            k_cell_dof (uint): The local dof index.
        """
        return self.get_dofs_coords(mesh, k_cell)[k_cell_dof]

    def _init_sym_dphi(self):
        """Computes the (symbolic) derivatives of the shape functions, and store them as a (n_dofs x dim) sympy Array."""
        assert (self.sym_phi.shape == (self.n_dofs,))
        assert (self.sym_x.shape == (self.dim,))
        self.sym_dphi = sympy.transpose(
            sympy.derive_by_array(
                self.sym_phi,
                self.sym_x))
        # self.sym_dphi = sympy.Array([[phi.diff(xi) for xi in self.sym_x] for phi in self.sym_phi])
        assert (self.sym_dphi.shape == (self.n_dofs, self.dim))

    def _init_sym_ddphi(self):
        """Computes the (symbolic) second derivatives of the shape functions, and store them as a (n_dofs x dim x dim) sympy Array."""
        assert (self.sym_dphi.shape == (self.n_dofs, self.dim))
        assert (self.sym_x.shape == (self.dim,))
        self.sym_ddphi = sympy.permutedims( # Generalization of transpose for high dimension arrays
            sympy.derive_by_array(
                self.sym_dphi,
                self.sym_x),
            (1, 2, 0))
        # self.sym_ddphi = sympy.Array([[[phi.diff(xi).diff(xj) for xj in self.sym_x] for xi in self.sym_x] for phi in self.sym_phi])
        assert (self.sym_ddphi.shape == (self.n_dofs, self.dim, self.dim))

    def _init_sym_phi_phi(self):
        """Computes the (symbolic) products of shape functions, and store them as a (n_dofs x n_dofs) sympy Array."""
        assert (self.sym_phi.shape == (self.n_dofs,))
        self.sym_phi_phi = sympy.tensorproduct(
            self.sym_phi,
            self.sym_phi)
        # self.sym_phi_phi = sympy.Array(self.sym_phi.tomatrix() * self.sym_phi.tomatrix().T)
        # self.sym_phi_phi = sympy.Array([[phik*phil for phil in self.sym_phi] for phik in self.sym_phi])
        assert (self.sym_phi_phi.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_dphi_dphi(self):
        """Computes the (symbolic) products of shape functions derivatives, and store them as a (n_dofs x n_dofs) sympy Array."""
        assert (self.sym_dphi.shape == (self.n_dofs, self.dim))
        self.sym_dphi_dphi = sympy.tensorcontraction(
            sympy.tensorproduct(
                self.sym_dphi,
                sympy.transpose(self.sym_dphi)),
            (1, 2))
        # self.sym_dphi_dphi = sympy.Array(self.sym_dphi.tomatrix() * self.sym_dphi.tomatrix().T)
        assert (self.sym_dphi_dphi.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_ddphi_ddphi(self):
        """Computes the (symbolic) products of shape functions second derivatives, and store them as a (n_dofs x n_dofs) sympy Array."""
        assert (self.sym_ddphi.shape == (self.n_dofs, self.dim, self.dim))
        self.sym_ddphi_ddphi = sympy.tensorcontraction(sympy.tensorcontraction(
            sympy.tensorproduct(
                self.sym_ddphi,
                sympy.permutedims( # Generalization of transpose for high dimension arrays
                    self.sym_ddphi,
                    (1, 2, 0))),
            (1, 3)), (1, 2))
        assert (self.sym_ddphi_ddphi.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_phi_dphi_dphi_phi(self):
        """Computes the (symbolic) products of shape functions and shape functions derivatives, and store them as a (n_dofs x n_dofs) sympy Array."""
        assert (self.sym_phi.shape == (self.n_dofs,))
        assert (self.sym_dphi.shape == (self.n_dofs, self.dim))
        self.sym_phi_dphi_dphi_phi = (sympy.tensorcontraction(
            sympy.tensorproduct(
                self.sym_phi,
                sympy.transpose(self.sym_dphi)),
            (1,)) + sympy.tensorcontraction(
            sympy.tensorproduct(
                self.sym_dphi,
                self.sym_phi), # MG20200609: No need to transpose sym_phi, which is a 1D array…
            (1,)))/2
        # self.sym_phi_dphi_dphi_phi = sympy.Array(self.sym_phi.tomatrix() * self.sym_dphi.tomatrix().T + self.sym_dphi.tomatrix() * self.sym_phi.tomatrix().T )/2
        assert (self.sym_phi_dphi_dphi_phi.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_phi_int(self, coeff=1, n=0):
        """Computes the (symbolic) integrals over the element of the shape functions (force vector)."""
        self.sym_phi_int = self._integrate_array(array=self.sym_phi, coeff=coeff, n=n)

    def _init_sym_phi_phi_int(self, coeff=1, n=0):
        """Computes the (symbolic) integrals over the element of the shape functions products (mass matrix)."""
        self.sym_phi_phi_int = self._integrate_array(array=self.sym_phi_phi, coeff=coeff, n=n)

    def _init_sym_dphi_dphi_int(self, coeff=1, n=0):
        """Computes the (symbolic) integrals over the element of the shape functions derivatives products (stiffness matrix)."""
        self.sym_dphi_dphi_int = self._integrate_array(array=self.sym_dphi_dphi, coeff=coeff, n=n)

    def _init_sym_ddphi_ddphi_int(self, coeff=1, n=0):
        """Computes the (symbolic) integrals over the element of the shape functions second derivatives products."""
        self.sym_ddphi_ddphi_int = self._integrate_array(array=self.sym_ddphi_ddphi, coeff=coeff, n=n)

    def _init_sym_phi_dphi_dphi_phi_int(self, coeff=1, n=0):
        """Computes the (symbolic) integral over the element of the shape functions and shape functions derivatives symmetrized products."""
        self.sym_phi_dphi_dphi_phi_int = self._integrate_array(array=self.sym_phi_dphi_dphi_phi, coeff=coeff, n=n)

    def init_get_phi_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions element integral."""
        self._init_sym_phi_int(coeff=coeff, n=n)
        self._get_phi_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_phi_int,
            modules="numpy")

    def get_phi_int(self, mesh, k_cell, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        loc_vec[:] = self._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))

    def init_get_phi_phi_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions products element integral."""
        self._init_sym_phi_phi()
        self._init_sym_phi_phi_int(coeff=coeff, n=n)
        self._get_phi_phi_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_phi_phi_int,
            modules="numpy")

    def get_phi_phi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions products element integral."""
        loc_mat[:,:] = self._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))

    def init_get_dphi_dphi_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions derivatives products element integral."""
        self._init_sym_dphi()
        self._init_sym_dphi_dphi()
        self._init_sym_dphi_dphi_int(coeff=coeff, n=n)
        self._get_dphi_dphi_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_dphi_dphi_int,
            modules="numpy")

    def get_dphi_dphi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions derivatives products element integral."""
        loc_mat[:,:] = self._get_dphi_dphi_int(*mesh.get_cell_nodes_coords(k_cell))

    def init_get_ddphi_ddphi_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions second derivatives products element integral."""
        self._init_sym_dphi()
        self._init_sym_ddphi()
        self._init_sym_ddphi_ddphi()
        self._init_sym_ddphi_ddphi_int(coeff=coeff, n=n)
        self._get_ddphi_ddphi_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_ddphi_ddphi_int,
            modules="numpy")

    def get_ddphi_ddphi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions second derivatives products element integral."""
        loc_mat[:,:] = self._get_ddphi_ddphi_int(*mesh.get_cell_nodes_coords(k_cell))

    def init_get_phi_dphi_dphi_phi_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions and shape function derivatives products element integral."""
        self._init_sym_dphi()
        self._init_sym_phi_dphi_dphi_phi()
        self._init_sym_phi_dphi_dphi_phi_int(coeff=coeff, n=n)
        self._get_phi_dphi_dphi_phi_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_phi_dphi_dphi_phi_int,
            modules="numpy")

    def get_phi_dphi_dphi_phi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of the shape functions and shape function derivatives products element integral."""
        loc_mat[:,:] = self._get_phi_dphi_dphi_phi_int(*mesh.get_cell_nodes_coords(k_cell))


class FiniteElement_1D(FiniteElement):
    def __init__(self):
        super().__init__() # FiniteElement.__init__()
        self.dim = 1
        self.sym_x = sympy.Array(sympy.symbols('x:{}'.format(self.dim)))


class FiniteElement_Line(FiniteElement_1D):
    def __init__(self):
        super().__init__() # FiniteElement_1D.__init__()
        self.shape = "Line"
        self.n_nodes = 2
        self.sym_nodes = sympy.Array(
            sympy.symbols('n:{}:{}'.format(self.n_nodes, self.dim)),
            (self.n_nodes, self.dim))
        self.n_edges = 0

    def _integrate_array(self, array, coeff=1, n=0):
        """
        Computes the (symbolic) integral over the element of a (1D or 2D symmetric) sympy array, and stores it as a sympy Array.
        The integrand can be multiplied by a (scalar) coefficent.
        The integrand can be multiplied by the first spatial variable to a given power, which is useful for instance in case of cylindrical or spherical symmetry.

        Args:
            coeff (float): The coefficient.
            n (uint): The power of the spatial variable.
        """
        array_int = sympy.MutableDenseNDimArray(array)
        if (array.rank() == 1):
            for i in range(array.shape[0]):
                array_int[i,] = sympy.integrate(
                    coeff * array[i] * self.sym_x[0]**n,
                    (self.sym_x[0], self.sym_nodes[0,0], self.sym_nodes[1,0]))
        elif (array.rank() == 2):
            assert (array.shape[1] == array.shape[0]), "Input array must be symmetric. Aborting."
            for i in range(0,array.shape[0]):
                for j in range(i,array.shape[0]):
                    array_int[i,j] = sympy.integrate(
                        coeff * array[i,j] * self.sym_x[0]**n,
                        (self.sym_x[0], self.sym_nodes[0,0], self.sym_nodes[1,0]))
            for i in range(0,array.shape[0]):
                for j in range(0,i):
                    array_int[i,j] = array_int[j,i]
        else:
            assert (0), "Input array must be 1D or 2D. Aborting."
        return (array_int)


class FiniteElement_Line_P0(FiniteElement_Line):
    def __init__(self):
        super().__init__() # FiniteElement_Line.__init__()
        self.interpolation = "P0"
        self.n_points = 1
        self.sym_points = sympy.Array([(self.sym_nodes[0]+self.sym_nodes[1])/2])
        self.n_dofs = 1
        self.dofs_attachement = ["cell"]
        self.dofs_attachement_idx = [0]
        self.sym_phi = sympy.Array([1.])


class FiniteElement_Line_P1(FiniteElement_Line):
    def __init__(self):
        super().__init__() # FiniteElement_Line.__init__()
        self.interpolation = "P1"
        self.n_points = 2
        self.sym_points = sympy.Array(
            [self.sym_nodes[0],
             self.sym_nodes[1]])
        self.n_dofs = 2
        self.dofs_attachement = ["node"]*2
        self.dofs_attachement_idx = [0, 1]
        self.sym_phi = sympy.Array(
            [(self.sym_points[1,0]-self.sym_x[0]       )/(self.sym_points[1,0]-self.sym_points[0,0]),
             (self.sym_x[0]       -self.sym_points[0,0])/(self.sym_points[1,0]-self.sym_points[0,0])])


class FiniteElement_Line_P2(FiniteElement_Line):
    def __init__(self):
        super().__init__() # FiniteElement_Line.__init__()
        self.interpolation = "P2"
        self.n_points = 3
        self.sym_points = sympy.Array(
            [self.sym_nodes[0], self.sym_nodes[1],
             self.sym_nodes[0] + (self.sym_nodes[1]-self.sym_nodes[0])/2])
        self.n_dofs = 3
        self.dofs_attachement = ["node"]*2 \
                              + ["cell"]
        self.dofs_attachement_idx = [0, 1, 0]
        self.sym_phi = sympy.Array(
            [(self.sym_x[0]-self.sym_points[1,0])/(self.sym_points[0,0]-self.sym_points[1,0]) * (self.sym_x[0]-self.sym_points[2,0])/(self.sym_points[0,0]-self.sym_points[2,0]),
             (self.sym_x[0]-self.sym_points[0,0])/(self.sym_points[1,0]-self.sym_points[0,0]) * (self.sym_x[0]-self.sym_points[2,0])/(self.sym_points[1,0]-self.sym_points[2,0]),
             (self.sym_x[0]-self.sym_points[0,0])/(self.sym_points[2,0]-self.sym_points[0,0]) * (self.sym_x[0]-self.sym_points[1,0])/(self.sym_points[2,0]-self.sym_points[1,0])])


class FiniteElement_Line_Pk(FiniteElement_Line):
    def __init__(self, order):
        super().__init__() # FiniteElement_Line.__init__()
        self.interpolation = "P"+str(order)
        self.n_points = order+1
        self.sym_points  = sympy.Array(
            [self.sym_nodes[0], self.sym_nodes[1]] \
          + [self.sym_nodes[0] + k_point * (self.sym_nodes[1]-self.sym_nodes[0])/(self.n_points-1) for k_point in range(1,self.n_points-1)])
        self.n_dofs = self.n_points
        self.dofs_attachement = ["node"]*2 \
                              + ["cell"]*(self.n_dofs-2)
        self.dofs_attachement_idx = [0, 1] \
                                  + [0]*(self.n_dofs-2)
        self.sym_phi = sympy.Array(
            [numpy.prod([(self.sym_x[0]-self.sym_points[l_dof,0])/(self.sym_points[k_dof,0]-self.sym_points[l_dof,0]) for l_dof in range(self.n_dofs) if l_dof != k_dof]) for k_dof in range(self.n_dofs)])


class FiniteElement_Line_H2k(FiniteElement_Line):
    def __init__(self, order):
        super().__init__() # FiniteElement_Line.__init__()
        self.interpolation = "H"+str(order)
        self.n_points = order+1
        self.sym_points  = sympy.Array(
            [self.sym_nodes[0], self.sym_nodes[1]] \
          + [self.sym_nodes[0] + k_point * (self.sym_nodes[1]-self.sym_nodes[0])/(self.n_points-1) for k_point in range(1,self.n_points-1)])
        self.n_dofs = 2*self.n_points
        self.dofs_attachement = ["node"]*4 \
                              + ["cell"]*(self.n_dofs-4)
        self.dofs_attachement_idx = [0, 0, 1, 1] \
                                  + [0]*(self.n_dofs-4)
        self.sym_Lk = [numpy.prod([(self.sym_x[0]-self.sym_points[l_point,0])/(self.sym_points[k_point,0]-self.sym_points[l_point,0]) for l_point in range(self.n_points) if l_point != k_point]) for k_point in range(self.n_points)]
        self.sym_phi = sympy.Array(
            [phik for k_point in range(self.n_points) for phik in [\
                (1-2*(self.sym_x[0]-self.sym_points[k_point,0]) * self.sym_Lk[k_point].diff(self.sym_x[0]).subs(self.sym_x[0],self.sym_points[k_point,0])) * self.sym_Lk[k_point]**2,
                (self.sym_x[0]-self.sym_points[k_point,0])*self.sym_Lk[k_point]**2]])


class FiniteElement_2D(FiniteElement):
    def __init__(self):
        super().__init__() # FiniteElement.__init__()
        self.dim = 2
        self.sym_x = sympy.Array(sympy.symbols('x:{}'.format(self.dim)))

    def _integrate_array(self, array, coeff=1, n=0):
        """
        Computes the (symbolic) integral over the element of a (1D or 2D symmetric) sympy array, and stores it as a sympy Array.
        The integrand can be multiplied by a (scalar) coefficent.
        The integrand can be multiplied by the first spatial variable to a given power, which is useful for instance in case of cylindrical or spherical symmetry.

        Args:
            coeff (float): The coefficient.
            n (uint): The power of the spatial variable.
        """
        array_int = sympy.MutableDenseNDimArray(array)
        if (array.rank() == 1):
            for i in range(array.shape[0]):
                array_int[i,] = 0.
                for pows, c in array[i].as_poly(self.sym_x[0], self.sym_x[1]).as_dict().items(): # MG20200501: polytope_integrate seems to require to integrate monomes alone…
                    array_int[i,] += coeff * c * polytope_integrate(
                        poly=self.sym_polygon,
                        expr=sympy.abc.x**(n+pows[0]) * sympy.abc.y**pows[1]) # MG20200511: polytope_integrate seems to require integration variable to be x, y, etc…
        elif (array.rank() == 2):
            assert (array.shape[1] == array.shape[0]), "Input array must be symmetric. Aborting."
            for i in range(0,array.shape[0]):
                for j in range(i,array.shape[0]):
                    array_int[i,j] = 0.
                    for pows, c in array[i,j].as_poly(self.sym_x[0], self.sym_x[1]).as_dict().items(): # MG20200501: polytope_integrate seems to require to integrate monomes alone…
                        array_int[i,j] += coeff * c * polytope_integrate(
                            poly=self.sym_polygon,
                            expr=sympy.abc.x**(n+pows[0]) * sympy.abc.y**pows[1]) # MG20200511: polytope_integrate seems to require integration variable to be x, y, etc.
            for i in range(0,array.shape[0]):
                for j in range(0,i):
                    array_int[i,j] = array_int[j,i]
        else:
            assert (0), "Input array must be 1D or 2D. Aborting."
        array_int /= sympy.sign(-self.sym_polygon.area) # MG20200501: Integral is positive for clockwise numbering, negative for counter-clockwise numbering…
        return (array_int)

    def _integrate_array_on_edges(self, array, coeff=1, n=0):
        """
        Computes the (symbolic) integral over the element edges of a (1D) sympy array, and stores it as a (1D) sympy Array.
        The integrand can be multiplied by a (scalar) coefficent.
        The integrand can be multiplied by the first spatial variable to a given power, which is useful for instance in case of cylindrical or spherical symmetry.

        Args:
            coeff (float): The coefficient.
            n (uint): The power of the spatial variable.
        """
        array_int = [sympy.MutableDenseNDimArray(array) for k_edge in range(self.n_edges)]
        if (array.rank() == 1):
            for k_edge in range(self.n_edges):
                for i in range(array.shape[0]):
                    array_int[k_edge][i,] = coeff * sympy.line_integrate(array[i] * self.sym_x[0]**n, self.sym_edges[k_edge], [self.sym_x[0], self.sym_x[1]])
        else:
            assert (0), "Input array must be 1D. Aborting."
        return (array_int)

    def _init_sym_phi_edge_int(self, coeff=1, n=0):
        """Computes the (symbolic) integrals over the element of the shape functions (force vector)."""
        self.sym_phi_edge_int = self._integrate_array_on_edges(array=self.sym_phi, coeff=coeff, n=n)

    def init_get_phi_edge_int(self, coeff=1, n=0):
        """Initializes the (efficient) computation of the shape functions edges integrals."""
        self._init_sym_phi_edge_int(coeff=coeff, n=n)
        self._get_phi_edge_int = [sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_phi_edge_int[k_edge],
            modules="numpy") \
        for k_edge in range(self.n_edges)]

    def get_phi_edge_int(self, mesh, k_cell, k_cell_edge, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        loc_vec[:] = self._get_phi_edge_int[k_cell_edge](*mesh.get_cell_nodes_coords(k_cell))


class FiniteElement_Triangle(FiniteElement_2D):
    def __init__(self):
        super().__init__() # FiniteElement_2D.__init__()
        self.shape = "Triangle"
        self.n_nodes = 3
        self.sym_nodes = sympy.Array(
            sympy.symbols('n:{}:{}'.format(self.n_nodes, self.dim)),
            (self.n_nodes, self.dim))
        self.sym_polygon = sympy.Polygon(
            sympy.Point(self.sym_nodes[0]),
            sympy.Point(self.sym_nodes[1]),
            sympy.Point(self.sym_nodes[2])) # This is used for element integral
        self.n_edges = 3
        self.sym_edges = [sympy.Curve((1-sympy.abc.t) * self.sym_nodes[k_edge] + sympy.abc.t * self.sym_nodes[(k_edge+1)%self.n_edges], (sympy.abc.t, 0, 1)) for k_edge in range(self.n_edges)] # This is used for edges integral


class FiniteElement_Triangle_P0(FiniteElement_Triangle):
    def __init__(self):
        super().__init__() # FiniteElement_Triangle.__init__()
        self.interpolation = "P0"
        self.n_points = 1
        self.sym_points = sympy.Array([(self.sym_nodes[0]+self.sym_nodes[1]+self.sym_nodes[2])/3])
        self.n_dofs = 1
        self.dofs_attachement = ["cell"]
        self.dofs_attachement_idx = [0]
        self.sym_phi = sympy.Array([1.])


def compute_Lagrange_shape_functions_through_linear_system(sym_x, sym_points):
    n_dofs = sym_points.shape[0]
    sym_phis = []
    for k_dof in range(n_dofs):
        # print (k_dof)
        ak = sympy.symbols('a:{}'.format(n_dofs))
        # print (ak)
        if   (n_dofs == 3):
            sym_phi = ak[0]            \
                    + ak[1] * sym_x[0] \
                    + ak[2] * sym_x[1]
        elif (n_dofs == 4):
            sym_phi = ak[0]                       \
                    + ak[1] * sym_x[0]            \
                    + ak[2]            * sym_x[1] \
                    + ak[3] * sym_x[0] * sym_x[1]
        elif (n_dofs == 6):
            sym_phi = ak[0]                             \
                    + ak[1] * sym_x[0]                  \
                    + ak[2]               * sym_x[1]    \
                    + ak[3] * sym_x[0]    * sym_x[1]    \
                    + ak[4] * sym_x[0]**2               \
                    + ak[5]               * sym_x[1]**2
        else:
            assert (0), "Not implemented. Aborting."
        # print (sym_phi)
        ak_sol = sympy.solve(
            [sym_phi.subs({sym_x[0]:sym_points[l_dof,0], sym_x[1]:sym_points[l_dof,1]}) - float(l_dof == k_dof) for l_dof in range(n_dofs)],
            ak)
        # print (ak_sol)
        sym_phi = sym_phi.subs(ak_sol)
        # print (sym_phi)
        sym_phis.append(sym_phi)
    return sym_phis


class FiniteElement_Triangle_P1(FiniteElement_Triangle):
    def __init__(self):
        super().__init__() # FiniteElement_Triangle.__init__()
        self.interpolation = "P1"
        self.n_points = 3
        self.sym_points = sympy.Array(
            [self.sym_nodes[0],
             self.sym_nodes[1],
             self.sym_nodes[2]])
        self.n_dofs = 3
        self.dofs_attachement = ["node"]*3
        self.dofs_attachement_idx = [0, 1, 2]
        self.sym_phi = sympy.Array(
            compute_Lagrange_shape_functions_through_linear_system(self.sym_x, self.sym_points))


class FiniteElement_Triangle_P2(FiniteElement_Triangle):
    def __init__(self):
        super().__init__() # FiniteElement_Triangle.__init__()
        self.interpolation = "P2"
        self.n_points = 6
        self.sym_points = sympy.Array(
            [self.sym_nodes[0],
             self.sym_nodes[1],
             self.sym_nodes[2],
            (self.sym_nodes[0]+self.sym_nodes[1])/2,
            (self.sym_nodes[1]+self.sym_nodes[2])/2,
            (self.sym_nodes[2]+self.sym_nodes[0])/2])
        self.n_dofs = 6
        self.dofs_attachement = ["node"]*3 \
                              + ["edge"]*3
        self.dofs_attachement_idx = [0, 1, 2, 0, 1, 2]
        self.sym_phi = sympy.Array(
            compute_Lagrange_shape_functions_through_linear_system(self.sym_x, self.sym_points))


class FiniteElement_Triangle_Pk(FiniteElement_Triangle):
    pass


class FiniteElement_Quadrangle(FiniteElement_2D):
    def __init__(self):
        super().__init__() # FiniteElement_2D.__init__()
        self.shape = "Quadrangle"
        self.n_nodes = 4
        self.sym_nodes = sympy.Array(
            sympy.symbols('n:{}:{}'.format(self.n_nodes, self.dim)),
            (self.n_nodes, self.dim))
        self.sym_polygon = sympy.Polygon(
            sympy.Point(self.sym_nodes[0]),
            sympy.Point(self.sym_nodes[1]),
            sympy.Point(self.sym_nodes[3]),
            sympy.Point(self.sym_nodes[2])) # This is used for element integral
        self.n_edges = 4
        self.sym_edges = [sympy.Curve((1-sympy.abc.t) * self.sym_nodes[k_edge] + sympy.abc.t * self.sym_nodes[(k_edge+1)%self.n_edges], (sympy.abc.t, 0, 1)) for k_edge in range(self.n_edges)] # This is used for edges integral


class FiniteElement_Quadrangle_Q0(FiniteElement_Quadrangle):
    def __init__(self):
        super().__init__() # FiniteElement_Quadrangle.__init__()
        self.interpolation = "Q0"
        self.n_points = 1
        self.sym_points = sympy.Array([(self.sym_nodes[0]+self.sym_nodes[1]+self.sym_nodes[2]+self.sym_nodes[3])/4])
        self.n_dofs = 1
        self.dofs_attachement = ["cell"]
        self.dofs_attachement_idx = [0]
        self.sym_phi = sympy.Array([1.])


class FiniteElement_Quadrangle_Q1(FiniteElement_Quadrangle):
    def __init__(self):
        super().__init__() # FiniteElement_Quadrangle.__init__()
        self.interpolation = "Q1"
        self.n_points = 4
        self.sym_points = sympy.Array(
            [self.sym_nodes[0],
             self.sym_nodes[1],
             self.sym_nodes[2],
             self.sym_nodes[3]])
        self.n_dofs = 4
        self.dofs_attachement = ["node"]*4
        self.dofs_attachement_idx = [0, 1, 2, 3]
#         self.sym_phi = sympy.Array(
#             [numpy.prod([(self.sym_x[0]-self.sym_points[l_dof,0])/(self.sym_points[k_dof,0]-self.sym_points[l_dof,0]) for l_dof in range(self.n_dofs) if l_dof != k_dof]) for k_dof in range(self.n_dofs)])
        self.sym_phi = sympy.Array(
            compute_Lagrange_shape_functions_through_linear_system(self.sym_x, self.sym_points))
        # self.sym_phi = sympy.Array(
        #     [(self.sym_points[1,0]-self.sym_x[0]       )/(self.sym_points[1,0]-self.sym_points[0,0])\
        #     *(self.sym_points[2,1]-self.sym_x[1]       )/(self.sym_points[2,1]-self.sym_points[0,1]),
        #      (self.sym_x[0]       -self.sym_points[0,0])/(self.sym_points[1,0]-self.sym_points[0,0])\
        #     *(self.sym_points[3,1]-self.sym_x[1]       )/(self.sym_points[3,1]-self.sym_points[1,1]),
        #      (self.sym_points[3,0]-self.sym_x[0]       )/(self.sym_points[3,0]-self.sym_points[2,0])\
        #     *(self.sym_x[1]       -self.sym_points[0,1])/(self.sym_points[2,1]-self.sym_points[0,1]),
        #      (self.sym_x[0]       -self.sym_points[2,0])/(self.sym_points[3,0]-self.sym_points[2,0])\
        #     *(self.sym_x[1]       -self.sym_points[1,1])/(self.sym_points[3,1]-self.sym_points[1,1])])


class FiniteElement_Quadrangle_Q2(FiniteElement_Quadrangle):
    pass


class FiniteElement_Quadrangle_Qk(FiniteElement_Quadrangle):
    pass


################################################################################


def create_finite_element(
        shape,
        interpolation,
        order=None):

    if (shape == "Line"):
        if   (interpolation == "P1"):
            return FiniteElement_Line_P1()
        elif (interpolation == "P2"):
            return FiniteElement_Line_P2()
        elif (interpolation == "Pk"):
            return FiniteElement_Line_Pk(order=order)
        else:
            assert (0), "Interpolation "+interpolation+" not implemented. Aborting."
    elif (shape == "Triangle"):
        if   (interpolation == "P1"):
            return FiniteElement_Triangle_P1()
        elif (interpolation == "P2"):
            return FiniteElement_Triangle_P2()
        elif (interpolation == "Pk"):
            return FiniteElement_Triangle_Pk(order=order)
        else:
            assert (0), "Interpolation "+interpolation+" not implemented. Aborting."
    else:
        assert (0), "Shape "+shape+" not implemented. Aborting."
