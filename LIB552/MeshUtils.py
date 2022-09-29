#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import numpy
import vtk
# import vtk.numpy_interface.dataset_adapter as dsa

import LIB552 as lib


################################################################################


def mesh_to_ugrid(mesh):
    """
    Converts a LIB552.Mesh into a VTK unstructured grid.
    Only works for regular triangles, quadrangles, tetrahedrons and hexahedrons.

    Args:
        mesh (LIB552.Mesh): The mesh.

    Returns:
        ugrid (vtkUnstructuredGrid): The unstructured grid.
    """

    ugrid = vtk.vtkUnstructuredGrid()
    coordinates = numpy.hstack((
        mesh.nodes,
        numpy.zeros([mesh.n_nodes, 3-mesh.dim]))) # vtk grids are always in 3D
    points = vtk.vtkPoints()
    points.SetData(vtk.util.numpy_support.numpy_to_vtk(coordinates))
    ugrid.SetPoints(points)
    # print("ugrid.GetNumberOfPoints() = "+str(ugrid.GetNumberOfPoints()))
    if   (mesh.cell.cell_type == "Triangle"):
        cell_vtk_type = vtk.VTK_TRIANGLE
        connectivity = numpy.hstack((
            numpy.full((mesh.n_cells, 1), mesh.cell.n_nodes, dtype=numpy.int64), # MG20201106: on Windows numpy.int returns int32, which numpy_to_vtkIdTypeArray does not like
            mesh.cells_nodes.astype(numpy.int64))).flatten() # MG20201106: on Windows numpy.int returns int32, which numpy_to_vtkIdTypeArray does not like
    elif (mesh.cell.cell_type == "Quadrangle"):
        cell_vtk_type = vtk.VTK_QUAD
        mesh.cells_nodes[:,[3,2]] = mesh.cells_nodes[:,[2,3]] # MG20200517: VTK does not use lexicographic ordering for quadrangles
        connectivity = numpy.hstack((
            numpy.full((mesh.n_cells, 1), mesh.cell.n_nodes, dtype=numpy.int64), # MG20201106: on Windows numpy.int returns int32, which numpy_to_vtkIdTypeArray does not like
            mesh.cells_nodes.astype(numpy.int64))).flatten() # MG20201106: on Windows numpy.int returns int32, which numpy_to_vtkIdTypeArray does not like
        mesh.cells_nodes[:,[3,2]] = mesh.cells_nodes[:,[2,3]] # MG20200517: VTK does not use lexicographic ordering for quadrangles
    else:
        assert (0), "Not implemented. Aborting."
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(mesh.n_cells, vtk.util.numpy_support.numpy_to_vtkIdTypeArray(connectivity))
    ugrid.SetCells(cell_vtk_type, cell_array)
    # print("ugrid.GetNumberOfCells() = "+str(ugrid.GetNumberOfCells()))
    # ugrid_np = dsa.WrapDataObject(ugrid)
    # print("ugrid_np.Points = "+str(ugrid_np.Points))
    # print("ugrid_np.Cells = "+str(ugrid_np.Cells))
    return ugrid


mesh_to_vtk = mesh_to_ugrid


def field_to_ugrid_isoparametric(field, mesh, field_name=None):
    """
    Converts a finite element (scalar or vector) field into a VTK unstructured grid with point data.
    Only works if all dofs are attached to nodes.
    It is assumed that the cells dofs connectivity matches the cells nodes connectivity.
    For vector fields, it is assumed that the dof ordering is point-wise.

    Args:
        field (numpy.ndarray of float): The field (n_dofs x 1).
        mesh (LIB552.Mesh): The mesh.

    Returns:
        ugrid (vtkUnstructuredGrid): The unstructured grid.
    """

    ugrid = lib.mesh_to_ugrid(mesh)
    n_dofs = len(field)
    field_dim = n_dofs//mesh.n_nodes
    # print(field_dim)
    if (field_dim == 1):
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            field.reshape((mesh.n_nodes, field_dim)))
    else:
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(
            numpy.hstack((
                field.reshape((mesh.n_nodes, field_dim)),
                numpy.zeros([mesh.n_nodes, 3-field_dim])))) # vtk vector fields are always in 3D
    # print(vtk_array)
    if (field_name is not None):
        vtk_array.SetName(field_name)
    if (field_dim == 1):
        ugrid.GetPointData().SetScalars(vtk_array)
    else:
        ugrid.GetPointData().SetScalars(vtk_array)
        ugrid.GetPointData().SetVectors(vtk_array)
    return ugrid


def field_to_ugrid(field, mesh, finite_element, dof_manager, field_name=None):
    """
    Converts a finite element (scalar or vector) field into a VTK unstructured grid with point data.
    Only works if all dofs are attached to nodes.

    Args:
        field (numpy.ndarray of float): The field (n_dofs x 1).
        mesh (LIB552.Mesh): The mesh.
        finite_element (LIB552.FiniteElement): The finite_element.
        dof_manager (LIB552.DofManager): The dof manager.

    Returns:
        ugrid (vtkUnstructuredGrid): The unstructured grid.
    """

    assert (all([(attachement == "node") for attachement in finite_element.dofs_attachement])) ,\
        "Not all dofs are attached to nodes. Aborting."
    ugrid = lib.mesh_to_ugrid(mesh)
    n_dofs = len(field)
    field_dim = n_dofs//mesh.n_nodes
    # print(field_dim)
    if (field_dim == 1):
        field_vtk = numpy.zeros((mesh.n_nodes, field_dim))
        for k_dof in range(n_dofs):
            k_cell, k_cell_dof = dof_manager.global_to_local[k_dof]
            k_cell_node = finite_element.dofs_attachement_idx[k_cell_dof]
            k_node = mesh.cells_nodes[k_cell, k_cell_node]
            field_vtk[k_node,0] = field[k_dof]
    elif (field_dim == 2):
        field_vtk = numpy.zeros((mesh.n_nodes, 3)) # vtk vector fields are always in 3D
        for k_dof in range(n_dofs):
            k_cell, k_cell_dof = dof_manager.global_to_local[k_dof]
            k_cell_node = finite_element.dofs_attachement_idx[k_cell_dof]
            k_node = mesh.cells_nodes[k_cell, k_cell_node]
            k_component = finite_element.dofs_component[k_cell_dof]
            field_vtk[k_node,k_component] = field[k_dof]
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(field_vtk)
    # print(vtk_array)
    if (field_name is not None):
        vtk_array.SetName(field_name)
    if (field_dim == 1):
        ugrid.GetPointData().SetScalars(vtk_array)
    else:
        ugrid.GetPointData().SetScalars(vtk_array)
        ugrid.GetPointData().SetVectors(vtk_array)
    return ugrid


def mesh_from_pygmsh(pygmsh_mesh):
    """
    Converts a pygmsh mesh to a LIB552 mesh.
    Only implemented for 2D meshes, made of triangles or quadrangles.

    Args:
        pygmsh_mesh (pygmsh.Mesh): The mesh in pygmsh format.

    Returns:
        mesh (LIB552.Mesh): The mesh in LIB552 format.
    """

    assert (numpy.allclose(pygmsh_mesh.points[:,2], 0)), "Mesh must be 2D. Aborting."
    dim = 2
    nodes = pygmsh_mesh.points[:,[0,1]]
    if   (pygmsh_mesh.cells[0].type == "triangle"):
        cell = lib.Cell_Triangle()
    elif (pygmsh_mesh.cells[0].type == "quad"):
        cell = lib.Cell_Quadrangle()
    else:
        assert (0), "Cells must be triangles or quandrangles. Aborting."
    cells_nodes = pygmsh_mesh.cells[0].data
    mesh = lib.Mesh(dim, nodes, cell, cells_nodes)
    return mesh


def mesh_from_gmsh(gmsh_mesh):
    """
    Converts a gmsh mesh to a LIB552 mesh.
    Only implemented for 2D meshes, made of triangles.

    Args:
        gmsh_mesh (gmsh.Mesh): The mesh in gmsh format.

    Returns:
        mesh (LIB552.Mesh): The mesh in LIB552 format.
    """

    nodes_ids, nodes_coords, nodes_params = gmsh_mesh.getNodes(dim=2, includeBoundary=True)
    n_nodes = len(nodes_ids)
    # print(n_nodes)
    # print(nodes_ids)
    # print(nodes_coords)
    nodes_coords = nodes_coords.reshape((n_nodes, 3))
    # print(nodes_coords)
    assert (numpy.allclose(nodes_coords[:,2], 0)), "Mesh must be 2D. Aborting."
    nodes_coords = nodes_coords[:,:2]
    # print(nodes_coords)

    elems_types, elems_ids, elems_nodes_ids = gmsh_mesh.getElements(dim=2)
    if (elems_types[0] == 2):
        cell = lib.Cell_Triangle()
    else:
        assert (0), "Cells must be triangles. Aborting."
    n_cells = len(elems_ids[0])
    # print(n_cells)
    cells_nodes_ids = elems_nodes_ids[0]
    # print(cells_nodes_ids)
    sorter = numpy.argsort(nodes_ids)
    cells_nodes_idx = sorter[numpy.searchsorted(nodes_ids, cells_nodes_ids, sorter=sorter)]
    # print(cells_nodes_idx)
    cells_nodes_idx = cells_nodes_idx.reshape((n_cells, cell.n_nodes))
    # print(cells_nodes_idx)

    mesh = lib.Mesh(dim=2, nodes=nodes_coords, cell=cell, cells_nodes=cells_nodes_idx)
    return mesh
