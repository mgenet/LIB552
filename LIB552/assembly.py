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


def assemble_vector(
        mesh,
        finite_element,
        get_loc_vec,
        dof_manager,
        vec=None):
    """
    Procedure to assemble a vector.
    Either you provide the array, in which case it is filled; if you do not provide the array, it is created, filled, and returned.

    Args:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.
        get_loc_vec (function): The function that computes the local/elementary vector.
        dof_manager (LIB552.DofManager): The dof manager.
        vec (numpy.array): The global vector (dof_manager.n_dofs) (if not provided, one will be created).

    Returns:
        vec (numpy.array): The global vector (dof_manager.n_dofs) (only returned if it was created within the function; if it was provided as input, it is filled in place and not returned.)
    """

    must_return = False
    if (vec is None):
        must_return = True
        vec = numpy.zeros(dof_manager.n_dofs)
    dofs_idx = numpy.empty(finite_element.n_dofs, dtype=numpy.uint)
    loc_vec = numpy.empty(finite_element.n_dofs)
    for cell_idx in range(mesh.n_cells):
        # print ("cell_idx = "+str(cell_idx))
        dofs_idx[:] = dof_manager.get_cell_dofs_idx(cell_idx)
        # print ("dofs_idx = "+str(dofs_idx))
        get_loc_vec(mesh=mesh, k_cell=cell_idx, loc_vec=loc_vec)
        # print ("loc_vec = "+str(loc_vec))
        vec[numpy.ix_(dofs_idx)] += loc_vec
    if (must_return):
        return vec

def assemble_vector_from_edge_integral(
        mesh,
        finite_element,
        get_loc_vec,
        dof_manager,
        imposed_edges_idx,
        vec=None):
    """
    Procedure to assemble a vector.
    Either you provide the array, in which case it is filled; if you do not provide the array, it is created, filled, and returned.

    Args:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.
        get_loc_vec (function): The function that computes the local/elementary vector.
        dof_manager (LIB552.DofManager): The dof manager.
        imposed_edges_idx (list of uints): List of edges on which to impose the force.
        vec (numpy.array): The global vector (dof_manager.n_dofs) (if not provided, one will be created).

    Returns:
        vec (numpy.array): The global vector (dof_manager.n_dofs) (only returned if it was created within the function; if it was provided as input, it is filled in place and not returned.)
    """

    must_return = False
    if (vec is None):
        must_return = True
        vec = numpy.zeros(dof_manager.n_dofs)
    dofs_glo_idx = numpy.empty(finite_element.n_dofs, dtype=numpy.uint)
    loc_vec = numpy.empty(finite_element.n_dofs)
    for cell_glo_idx in range(mesh.n_cells):
        # print ("cell_glo_idx = "+str(cell_glo_idx))
        dofs_glo_idx[:] = dof_manager.local_to_global[cell_glo_idx]
        # print ("dofs_glo_idx = "+str(dofs_glo_idx))
        for edge_loc_idx in range(mesh.cell.n_edges):
            edge_glo_idx = mesh.cells_edges[cell_glo_idx, edge_loc_idx]
            if (edge_glo_idx in imposed_edges_idx):
                # print ("cell_glo_idx = "+str(cell_glo_idx))
                # print ("dofs_glo_idx = "+str(dofs_glo_idx))
                get_loc_vec(mesh=mesh, k_cell=cell_glo_idx, k_cell_edge=edge_loc_idx, loc_vec=loc_vec)
                # print ("loc_vec = "+str(loc_vec))
                vec[numpy.ix_(dofs_glo_idx)] += loc_vec
    if (must_return):
        return vec

def assemble_matrix(
        mesh,
        finite_element,
        get_loc_mat,
        dof_manager,
        mat=None):
    """
    Procedure to assemble a matrix.
    Either you provide the array, in which case it is filled; if you do not provide the array, it is created, filled, and returned.

    Args:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.
        get_loc_mat (function): The function that computes the local/elementary matrix.
        dof_manager (LIB552.DofManager): The dof manager.
        mat (numpy.array): The global matrix (dof_manager.n_dofs x dof_manager.n_dofs) (if not provided, one will be created).

    Returns:
        mat (numpy.array): The global matrix (dof_manager.n_dofs x dof_manager.n_dofs) (only returned if it was created within the function; if it was provided as input, it is filled in place and not returned.)
    """

    must_return = False
    if (mat is None):
        must_return = True
        mat = numpy.zeros((dof_manager.n_dofs,dof_manager.n_dofs))
    dofs_idx = numpy.empty(finite_element.n_dofs, dtype=numpy.uint)
    loc_mat = numpy.empty((finite_element.n_dofs,finite_element.n_dofs))
    for cell_idx in range(mesh.n_cells):
        # print ("cell_idx = "+str(cell_idx))
        dofs_idx[:] = dof_manager.get_cell_dofs_idx(cell_idx)
        # print ("dofs_idx = "+str(dofs_idx))
        get_loc_mat(mesh=mesh, k_cell=cell_idx, loc_mat=loc_mat)
        # print ("loc_mat = "+str(loc_mat))
        mat[numpy.ix_(dofs_idx,dofs_idx)] += loc_mat
    if (must_return):
        return mat

def assemble_system_w_constraints(
        mesh,
        finite_element,
        get_loc_mat,
        get_loc_vec,
        dof_manager,
        prescribed_dofs_idx=[],
        prescribed_dofs_vals=[],
        mat=None,
        vec=None):
    """
    Procedure to assemble a system.
    Either you provide the arrays, in which case they are filled; if you do not provide the arrays, they are created, filled, and returned.
    Constraints are taken into account while keeping the matrix symmetric.

    Args:
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite element.
        get_loc_mat (function): The function that computes the local/elementary matrix.
        get_loc_vec (function): The function that computes the local/elementary vector.
        dof_manager (LIB552.DofManager): The dof manager.
        prescribed_dofs_idx (list of uints): List of constrained dofs indexes.
        prescribed_dofs_vals (list of floats): List of constrained dofs values.
        mat (numpy.array): The global matrix (dof_manager.n_dofs x dof_manager.n_dofs) (if not provided, one will be created).
        vec (numpy.array): The global vector (dof_manager.n_dofs) (if not provided, one will be created).

    Returns:
        mat (numpy.array): The global matrix (dof_manager.n_dofs x dof_manager.n_dofs) (only returned if it was created within the function; if it was provided as input, it is filled in place and not returned).
        vec (numpy.array): The global vector (dof_manager.n_dofs) (only returned if it was created within the function; if it was provided as input, it is filled in place and not returned).
    """

    must_return = False
    if (mat is None):
        must_return = True
        mat = numpy.zeros((dof_manager.n_dofs, dof_manager.n_dofs))
    if (vec is None):
        must_return = True
        vec = numpy.zeros(dof_manager.n_dofs)
    dofs_idx = numpy.empty(finite_element.n_dofs, dtype=numpy.uint)
    cell_prescribed_dofs_idx = numpy.empty(finite_element.n_dofs, dtype=numpy.int) # MG 20201031: Useless, right?
    loc_mat = numpy.empty((finite_element.n_dofs, finite_element.n_dofs))
    loc_vec = numpy.empty((finite_element.n_dofs))
    for cell_idx in range(mesh.n_cells):
        # print ("cell_idx = "+str(cell_idx))
        dofs_idx[:] = dof_manager.get_cell_dofs_idx(cell_idx)
        get_loc_mat(mesh=mesh, k_cell=cell_idx, loc_mat=loc_mat)
        get_loc_vec(mesh=mesh, k_cell=cell_idx, loc_vec=loc_vec)
        # print ("loc_mat = "+str(loc_mat))
        # print ("loc_vec = "+str(loc_vec))
        prescribed_dofs_loc_idx = []
        cell_prescribed_dofs_idx = []
        cell_prescribed_dofs_vals = []
        unprescribed_dofs_loc_idx = []
        for dof_loc_idx in range(finite_element.n_dofs):
            # print ("dof_loc_idx = "+str(dof_loc_idx))
            dof_idx = dofs_idx[dof_loc_idx]
            prescribed_dof_idx = numpy.argwhere(prescribed_dofs_idx == dof_idx)
            if (len(prescribed_dof_idx) > 0):
                prescribed_dof_idx = prescribed_dof_idx[0][0]
                prescribed_dofs_loc_idx.append(dof_loc_idx)
                cell_prescribed_dofs_idx.append(prescribed_dofs_idx[prescribed_dof_idx])
                cell_prescribed_dofs_vals.append(prescribed_dofs_vals[prescribed_dof_idx])
            else:
                unprescribed_dofs_loc_idx.append(dof_loc_idx)
        # print ("prescribed_dofs_loc_idx = "+str(prescribed_dofs_loc_idx))
        # print ("cell_prescribed_dofs_idx = "+str(cell_prescribed_dofs_idx))
        # print ("cell_prescribed_dofs_vals = "+str(cell_prescribed_dofs_vals))
        # print ("unprescribed_dofs_loc_idx = "+str(unprescribed_dofs_loc_idx))
        for dof_loc_idx, dof_val in zip(prescribed_dofs_loc_idx, cell_prescribed_dofs_vals):
            # print ("dof_loc_idx = "+str(dof_loc_idx))
            # print ("dof_val = "+str(dof_val))
            loc_mat[dof_loc_idx, :] = 0.
            loc_mat[dof_loc_idx, dof_loc_idx] = 1.
            loc_vec[dof_loc_idx] = dof_val
            # print ("loc_mat = "+str(loc_mat))
            # print ("loc_vec = "+str(loc_vec))
        for dof_loc_idx in unprescribed_dofs_loc_idx:
            # print ("dof_loc_idx = "+str(dof_loc_idx))
            loc_vec[dof_loc_idx] -= numpy.dot(loc_mat[dof_loc_idx, prescribed_dofs_loc_idx], cell_prescribed_dofs_vals)
            loc_mat[dof_loc_idx, prescribed_dofs_loc_idx] = 0.
            # print ("loc_mat = "+str(loc_mat))
            # print ("loc_vec = "+str(loc_vec))
        # print ("loc_mat = "+str(loc_mat))
        # print ("loc_vec = "+str(loc_vec))
        mat[numpy.ix_(dofs_idx, dofs_idx)] += loc_mat
        vec[numpy.ix_(dofs_idx)] += loc_vec
    if (must_return):
        return mat, vec
