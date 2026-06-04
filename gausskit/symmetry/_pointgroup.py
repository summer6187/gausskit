"""Molecular point-group analysis (vendored, trimmed from pymatgen).

This module is a trimmed copy of pymatgen's molecular point-group machinery,
used here only to compute the external rotational symmetry number of a molecule
from its geometry. It is adapted from materialsproject/pymatgen-core
(commit 7ae5068), files:

    src/pymatgen/symmetry/analyzer.py   (PointGroupAnalyzer, cluster_sites,
                                         generate_full_symmops)
    src/pymatgen/core/operations.py     (SymmOp)
    src/pymatgen/util/coord.py          (find_in_coord_list)

Only the molecular path is kept: ``SymmOp`` is reduced to the rotation/reflection
constructors actually used, and ``PointGroupAnalyzer`` drops the crystal
``SpacegroupAnalyzer`` and the atom-equivalence / symmetrize methods. The class
operates on a duck-typed molecule (a sites list with ``.coords`` / ``.species``
and ``.get_centered_molecule()`` / ``.cart_coords``); the gausskit adapter that
wraps an ``ase.Atoms`` lives in ``gausskit.symmetry`` (``__init__.py``).

NOTE on tolerances: the public wrapper pins ``tolerance=0.2`` and
``eigen_tolerance=1e-4`` rather than pymatgen's defaults (0.3, 0.01). Highly
elongated van-der-Waals complexes (e.g. a barrierless entrance channel with a
5 A separation) have a tiny long-axis moment of inertia, so the normalized
eigenvalue product falls below the 0.01 default and the complex is misclassified
as *linear* (sigma=1) before its C2 axis is tested. eigen_tolerance=1e-4 fixes
this while still flagging genuine linear molecules (whose product is ~0).

pymatgen is released under the MIT License:

    Copyright (c) 2011-2012 MIT & The Regents of the University of California,
    through Lawrence Berkeley National Laboratory

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy.cluster.hierarchy
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


class SymmOp:
    """A symmetry operation in Cartesian space, stored as a 4x4 affine matrix.

    Trimmed from pymatgen.core.operations.SymmOp: only the constructors and the
    ``operate`` / ``rotation_matrix`` accessors used by PointGroupAnalyzer are kept
    (the MSONable base, tensor/vector helpers and xyz-string I/O are dropped).
    """

    def __init__(self, affine_transformation_matrix: ArrayLike, tol: float = 0.01) -> None:
        affine_transformation_matrix = np.asarray(affine_transformation_matrix)
        shape = affine_transformation_matrix.shape
        if shape != (4, 4):
            raise ValueError(f"Affine Matrix must be a 4x4 numpy array, got {shape=}")
        self.affine_matrix = affine_transformation_matrix
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self.affine_matrix, other.affine_matrix, atol=self.tol)

    def __hash__(self) -> int:
        return 7

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation_matrix: ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        translation_vec: ArrayLike = (0, 0, 0),
        tol: float = 0.1,
    ) -> SymmOp:
        """Create a symmetry operation from a rotation matrix and a translation vector."""
        rotation_matrix = np.asarray(rotation_matrix)
        translation_vec = np.asarray(translation_vec)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation Matrix must be a 3x3 numpy array.")
        if translation_vec.shape != (3,):
            raise ValueError("Translation vector must be a rank 1 numpy array with 3 elements.")
        affine_matrix = np.eye(4)
        affine_matrix[:3][:, :3] = rotation_matrix
        affine_matrix[:3][:, 3] = translation_vec
        return cls(affine_matrix, tol)

    def operate(self, point: ArrayLike) -> NDArray:
        """Apply the operation on a point."""
        affine_point = np.append(point, 1.0)
        return np.dot(self.affine_matrix, affine_point)[:3]

    @property
    def rotation_matrix(self) -> NDArray:
        """A 3x3 numpy.array representing the rotation matrix."""
        return self.affine_matrix[:3][:, :3]

    @staticmethod
    def from_axis_angle_and_translation(
        axis: NDArray,
        angle: float,
        angle_in_radians: bool = False,
        translation_vec: Sequence[float] | NDArray = (0, 0, 0),
    ) -> SymmOp:
        """Generate a SymmOp for a rotation about a given axis plus translation."""
        if isinstance(axis, (tuple, list)):
            axis = np.array(axis)
        vec = np.asarray(translation_vec)
        ang = angle if angle_in_radians else angle * np.pi / 180
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        unit_vec = axis / np.linalg.norm(axis)
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = cos_a + unit_vec[0] ** 2 * (1 - cos_a)
        rot_mat[0, 1] = unit_vec[0] * unit_vec[1] * (1 - cos_a) - unit_vec[2] * sin_a
        rot_mat[0, 2] = unit_vec[0] * unit_vec[2] * (1 - cos_a) + unit_vec[1] * sin_a
        rot_mat[1, 0] = unit_vec[0] * unit_vec[1] * (1 - cos_a) + unit_vec[2] * sin_a
        rot_mat[1, 1] = cos_a + unit_vec[1] ** 2 * (1 - cos_a)
        rot_mat[1, 2] = unit_vec[1] * unit_vec[2] * (1 - cos_a) - unit_vec[0] * sin_a
        rot_mat[2, 0] = unit_vec[0] * unit_vec[2] * (1 - cos_a) - unit_vec[1] * sin_a
        rot_mat[2, 1] = unit_vec[1] * unit_vec[2] * (1 - cos_a) + unit_vec[0] * sin_a
        rot_mat[2, 2] = cos_a + unit_vec[2] ** 2 * (1 - cos_a)
        return SymmOp.from_rotation_and_translation(rot_mat, vec)

    @staticmethod
    def from_origin_axis_angle(
        origin: Sequence[float] | NDArray,
        axis: Sequence[float] | NDArray,
        angle: float,
        angle_in_radians: bool = False,
    ) -> SymmOp:
        """Generate a SymmOp for a rotation about a given axis through an origin."""
        theta = angle if angle_in_radians else angle * np.pi / 180
        a, b, c = origin
        ax_u, ax_v, ax_w = axis
        u2, v2, w2 = ax_u * ax_u, ax_v * ax_v, ax_w * ax_w
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        l2 = u2 + v2 + w2
        lsqrt = math.sqrt(l2)
        m11 = (u2 + (v2 + w2) * cos_t) / l2
        m12 = (ax_u * ax_v * (1 - cos_t) - ax_w * lsqrt * sin_t) / l2
        m13 = (ax_u * ax_w * (1 - cos_t) + ax_v * lsqrt * sin_t) / l2
        m14 = (
            a * (v2 + w2)
            - ax_u * (b * ax_v + c * ax_w)
            + (ax_u * (b * ax_v + c * ax_w) - a * (v2 + w2)) * cos_t
            + (b * ax_w - c * ax_v) * lsqrt * sin_t
        ) / l2
        m21 = (ax_u * ax_v * (1 - cos_t) + ax_w * lsqrt * sin_t) / l2
        m22 = (v2 + (u2 + w2) * cos_t) / l2
        m23 = (ax_v * ax_w * (1 - cos_t) - ax_u * lsqrt * sin_t) / l2
        m24 = (
            b * (u2 + w2)
            - ax_v * (a * ax_u + c * ax_w)
            + (ax_v * (a * ax_u + c * ax_w) - b * (u2 + w2)) * cos_t
            + (c * ax_u - a * ax_w) * lsqrt * sin_t
        ) / l2
        m31 = (ax_u * ax_w * (1 - cos_t) - ax_v * lsqrt * sin_t) / l2
        m32 = (ax_v * ax_w * (1 - cos_t) + ax_u * lsqrt * sin_t) / l2
        m33 = (w2 + (u2 + v2) * cos_t) / l2
        m34 = (
            c * (u2 + v2)
            - ax_w * (a * ax_u + b * ax_v)
            + (ax_w * (a * ax_u + b * ax_v) - c * (u2 + v2)) * cos_t
            + (a * ax_v - b * ax_u) * lsqrt * sin_t
        ) / l2
        return SymmOp(
            np.array(
                [
                    [m11, m12, m13, m14],
                    [m21, m22, m23, m24],
                    [m31, m32, m33, m34],
                    [0, 0, 0, 1],
                ]
            )
        )

    @staticmethod
    def reflection(normal: ArrayLike, origin: ArrayLike = (0, 0, 0)) -> SymmOp:
        """Get reflection symmetry operation about the plane with the given normal."""
        u, v, w = np.array(normal, dtype=float) / np.linalg.norm(normal)
        translation = np.eye(4)
        translation[:3, 3] = -np.asarray(origin)
        xx = 1 - 2 * u**2
        yy = 1 - 2 * v**2
        zz = 1 - 2 * w**2
        xy = -2 * u * v
        xz = -2 * u * w
        yz = -2 * v * w
        mirror_mat = np.array([[xx, xy, xz, 0], [xy, yy, yz, 0], [xz, yz, zz, 0], [0, 0, 0, 1]])
        if np.linalg.norm(origin) > 1e-6:
            mirror_mat = np.dot(np.linalg.inv(translation), np.dot(mirror_mat, translation))
        return SymmOp(mirror_mat)

    @staticmethod
    def inversion(origin: ArrayLike = (0, 0, 0)) -> SymmOp:
        """Inversion symmetry operation about the origin."""
        mat = -np.eye(4)
        mat[3, 3] = 1
        mat[:3, 3] = 2 * np.asarray(origin)
        return SymmOp(mat)

    @staticmethod
    def rotoreflection(axis: Sequence[float], angle: float, origin: Sequence[float] = (0, 0, 0)) -> SymmOp:
        """Get a roto-reflection symmetry operation."""
        rot = SymmOp.from_origin_axis_angle(origin, axis, angle)
        refl = SymmOp.reflection(axis, origin)
        matrix = np.dot(rot.affine_matrix, refl.affine_matrix)
        return SymmOp(matrix)


def find_in_coord_list(coord_list, coord, atol: float = 1e-8):
    """Find the indices of matches of a particular coord in a coord_list."""
    if len(coord_list) == 0:
        return []
    diff = np.array(coord_list) - np.array(coord)[None, :]
    return np.where(np.all(np.abs(diff) < atol, axis=1))[0]


def cluster_sites(
    mol: Molecule,
    tol: float,
    give_only_index: bool = False,
) -> tuple[int | Site | None, dict]:
    """Cluster sites based on distance and species type.

    Args:
        mol (Molecule): Molecule **with origin at center of mass**.
        tol (float): Tolerance to use.
        give_only_index (bool): Whether to return only the index of the
            origin site, instead of the site itself. Defaults to False.

    Returns:
        tuple[Site | None, dict]: origin_site is a site at the center
            of mass (None if there are no origin atoms). clustered_sites is a
            dict of {(avg_dist, species_and_occu): [list of sites]}
    """
    # Cluster works for dim > 2 data. We just add a dummy 0 for second
    # coordinate.
    dists: list[list[float]] = [[float(np.linalg.norm(site.coords)), 0] for site in mol]

    f_cluster = scipy.cluster.hierarchy.fclusterdata(dists, tol, criterion="distance")
    clustered_dists: dict[str, list[list[float]]] = defaultdict(list)
    for idx in range(len(mol)):
        clustered_dists[f_cluster[idx]].append(dists[idx])
    avg_dist = {key: np.mean(val) for key, val in clustered_dists.items()}
    clustered_sites = defaultdict(list)
    origin_site = None
    for idx, site in enumerate(mol):
        if avg_dist[f_cluster[idx]] < tol:
            origin_site = idx if give_only_index else site
        elif give_only_index:
            clustered_sites[avg_dist[f_cluster[idx]], site.species].append(idx)
        else:
            clustered_sites[avg_dist[f_cluster[idx]], site.species].append(site)
    return origin_site, clustered_sites


def generate_full_symmops(
    symmops: Sequence[SymmOp],
    tol: float,
) -> Sequence[SymmOp]:
    """Recursive algorithm to permute through all possible combinations of the initially
    supplied symmetry operations to arrive at a complete set of operations mapping a
    single atom to all other equivalent atoms in the point group. This assumes that the
    initial number already uniquely identifies all operations.

    Args:
        symmops (list[SymmOp]): Initial set of symmetry operations.
        tol (float): Tolerance for detecting symmetry.

    Returns:
        list[SymmOp]: Full set of symmetry operations.
    """
    # Uses an algorithm described in:
    # Gregory Butler. Fundamental Algorithms for Permutation Groups.
    # Lecture Notes in Computer Science (Book 559). Springer, 1991. page 15
    identity = np.eye(4)
    generators = [op.affine_matrix for op in symmops if not np.allclose(op.affine_matrix, identity)]
    if not generators:
        # C1 symmetry breaks assumptions in the algorithm afterwards
        return symmops

    full = list(generators)

    for g in full:
        for s in generators:
            op = np.dot(g, s)
            d = np.abs(full - op) < tol
            if not np.any(np.all(np.all(d, axis=2), axis=1)):
                full.append(op)
            if len(full) > 1000:
                warnings.warn(
                    f"{len(full)} matrices have been generated. The tol may be too small. Please terminate"
                    " and rerun with a different tolerance.",
                    stacklevel=2,
                )

    d = np.abs(full - identity) < tol
    if not np.any(np.all(np.all(d, axis=2), axis=1)):
        full.append(identity)
    return [SymmOp(op) for op in full]


class PointGroupAnalyzer:
    """A class to analyze the point group of a molecule.

    The general outline of the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.

        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymmetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis. All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.

    Attribute:
        sch_symbol (str): Schoenflies symbol of the detected point group.
    """

    inversion_op = SymmOp.inversion()

    def __init__(
        self,
        mol: Molecule,
        tolerance: float = 0.3,
        eigen_tolerance: float = 0.01,
        matrix_tolerance: float = 0.1,
    ) -> None:
        """The default settings are usually sufficient.

        Args:
            mol (Molecule): Molecule to determine point group for.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            eigen_tolerance (float): Tolerance to compare eigen values of
                the inertia tensor. Defaults to 0.01.
            matrix_tolerance (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        self.mol = mol
        self.centered_mol = mol.get_centered_molecule()
        self.tol = tolerance
        self.eig_tol = eigen_tolerance
        self.mat_tol = matrix_tolerance
        self._analyze()
        if self.sch_symbol in {"C1v", "C1h"}:
            self.sch_symbol: str = "Cs"

    def _analyze(self) -> None:
        if len(self.centered_mol) == 1:
            self.sch_symbol = "Kh"
        else:
            inertia_tensor = np.zeros((3, 3))
            total_inertia = 0
            for site in self.centered_mol:
                c = site.coords
                wt = site.species.weight
                for i in range(3):
                    inertia_tensor[i, i] += wt * (c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2)
                for i, j in ((0, 1), (1, 2), (0, 2)):
                    inertia_tensor[i, j] += -wt * c[i] * c[j]
                    inertia_tensor[j, i] += -wt * c[j] * c[i]
                total_inertia += wt * np.dot(c, c)

            # Normalize the inertia tensor so that it does not scale with size
            # of the system. This mitigates the problem of choosing a proper
            # comparison tolerance for the eigenvalues.
            inertia_tensor /= total_inertia
            eigvals, eigvecs = np.linalg.eig(inertia_tensor)
            self.principal_axes = eigvecs.T
            self.eigvals = eigvals
            v1, v2, v3 = eigvals
            eig_zero = abs(v1 * v2 * v3) < self.eig_tol
            eig_all_same = abs(v1 - v2) < self.eig_tol and abs(v1 - v3) < self.eig_tol
            eig_all_diff = abs(v1 - v2) > self.eig_tol and abs(v1 - v3) > self.eig_tol and abs(v2 - v3) > self.eig_tol

            self.rot_sym: list = []
            self.symmops: list[SymmOp] = [SymmOp(np.eye(4))]
            if eig_zero:
                logger.debug("Linear molecule detected")
                self._proc_linear()
            elif eig_all_same:
                logger.debug("Spherical top molecule detected")
                self._proc_sph_top()
            elif eig_all_diff:
                logger.debug("Asymmetric top molecule detected")
                self._proc_asym_top()
            else:
                logger.debug("Symmetric top molecule detected")
                self._proc_sym_top()

    def _proc_linear(self) -> None:
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = "D*h"
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            self.sch_symbol = "C*v"

    def _proc_asym_top(self) -> None:
        """Handles asymmetric top molecules, which cannot contain rotational symmetry
        larger than 2.
        """
        self._check_R2_axes_asym()
        if len(self.rot_sym) == 0:
            logger.debug("No rotation symmetries detected.")
            self._proc_no_rot_sym()
        elif len(self.rot_sym) == 3:
            logger.debug("Dihedral group detected.")
            self._proc_dihedral()
        else:
            logger.debug("Cyclic group detected.")
            self._proc_cyclic()

    def _proc_sym_top(self) -> None:
        """Handles symmetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.

        More complex handling required to look for R2 axes perpendicular to this unique
        axis.
        """
        if abs(self.eigvals[0] - self.eigvals[1]) < self.eig_tol:
            ind = 2
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.eig_tol:
            ind = 0
        else:
            ind = 1
        logger.debug(f"Eigenvalues = {self.eigvals}.")
        unique_axis = self.principal_axes[ind]
        self._check_rot_sym(unique_axis)
        logger.debug(f"Rotation symmetries = {self.rot_sym}")
        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(unique_axis)

        if len(self.rot_sym) >= 2:
            self._proc_dihedral()
        elif len(self.rot_sym) == 1:
            self._proc_cyclic()
        else:
            self._proc_no_rot_sym()

    def _proc_no_rot_sym(self) -> None:
        """Handles molecules with no rotational symmetry.

        Only possible point groups are C1, Cs and Ci.
        """
        self.sch_symbol = "C1"
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = "Ci"
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            for v in self.principal_axes:
                mirror_type = self._find_mirror(v)
                if mirror_type != "":
                    self.sch_symbol = "Cs"
                    break

    def _proc_cyclic(self) -> None:
        """Handles cyclic group molecules."""
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f"C{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type == "v":
            self.sch_symbol += "v"
        elif mirror_type == "" and self.is_valid_op(SymmOp.rotoreflection(main_axis, angle=180 / rot)):
            self.sch_symbol = f"S{2 * rot}"

    def _proc_dihedral(self) -> None:
        """Handles dihedral group molecules, i.e those with intersecting R2 axes and a
        main axis.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f"D{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type != "":
            self.sch_symbol += "d"

    def _check_R2_axes_asym(self) -> None:
        """Test for 2-fold rotation along the principal axes.

        Used to handle asymmetric top molecules.
        """
        for v in self.principal_axes:
            op = SymmOp.from_axis_angle_and_translation(v, 180)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((v, 2))

    def _find_mirror(self, axis: NDArray) -> Literal["h", "d", "v", ""]:
        """Looks for mirror symmetry of specified type about axis.

        Possible types are "h" or "vd". Horizontal (h) mirrors are perpendicular to the
        axis while vertical (v) or diagonal (d) mirrors are parallel. v mirrors has atoms
        lying on the mirror plane while d mirrors do not.
        """
        mirror_type: Literal["h", "d", "v", ""] = ""

        # First test whether the axis itself is the normal to a mirror plane.
        if self.is_valid_op(SymmOp.reflection(axis)):
            self.symmops.append(SymmOp.reflection(axis))
            mirror_type = "h"
        else:
            # Iterate through all pairs of atoms to find mirror
            for s1, s2 in itertools.combinations(self.centered_mol, 2):
                if s1.species == s2.species:
                    normal = s1.coords - s2.coords
                    if np.dot(normal, axis) < self.tol:
                        op = SymmOp.reflection(normal)
                        if self.is_valid_op(op):
                            self.symmops.append(op)
                            if len(self.rot_sym) > 1:
                                mirror_type = "d"
                                for v, _ in self.rot_sym:
                                    if np.linalg.norm(v - axis) >= self.tol and np.dot(v, normal) < self.tol:
                                        mirror_type = "v"
                                        break
                            else:
                                mirror_type = "v"
                            break

        return mirror_type

    def _get_smallest_set_not_on_axis(self, axis: NDArray) -> list:
        """Get the smallest list of atoms with the same species and distance from
        origin AND does not lie on the specified axis.

        This maximal set limits the possible rotational symmetry operations, since atoms
        lying on a test axis is irrelevant in testing rotational symmetryOperations.
        """

        def not_on_axis(site):
            return np.linalg.norm(np.cross(site.coords, axis)) > self.tol

        valid_sets = []
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        for test_set in dist_el_sites.values():
            valid_set = list(filter(not_on_axis, test_set))
            if len(valid_set) > 0:
                valid_sets.append(valid_set)

        return min(valid_sets, key=len)

    def _check_rot_sym(self, axis: NDArray) -> int:
        """Determine the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        max_sym = len(min_set)
        for idx in range(max_sym, 0, -1):
            if max_sym % idx != 0:
                continue
            op = SymmOp.from_axis_angle_and_translation(axis, 360 / idx)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((axis, idx))
                return idx
        return 1

    def _check_perpendicular_r2_axis(self, axis: NDArray) -> None | Literal[True]:
        """Check for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        for s1, s2 in itertools.combinations(min_set, 2):
            test_axis = np.cross(s1.coords - s2.coords, axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                if self.is_valid_op(op):
                    self.symmops.append(op)
                    self.rot_sym.append((test_axis, 2))
                    return True
        return None

    def _proc_sph_top(self) -> None:
        """Handles Spherical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self._find_spherical_axes()
        if len(self.rot_sym) == 0:
            logger.debug("Accidental spherical top!")
            self._proc_sym_top()
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        if rot < 3:
            logger.debug("Accidental spherical top!")
            self._proc_sym_top()

        elif rot == 3:
            mirror_type = self._find_mirror(main_axis)
            if mirror_type == "":
                self.sch_symbol = "T"
            elif self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Th"
            else:
                self.sch_symbol = "Td"

        elif rot == 4:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Oh"
            else:
                self.sch_symbol = "O"

        elif rot == 5:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Ih"
            else:
                self.sch_symbol = "I"

    def _find_spherical_axes(self) -> None:
        """Looks for R5, R4, R3 and R2 axes in spherical top molecules.

        Point group T molecules have only one unique 3-fold and one unique 2-fold axis. O
        molecules have one unique 4, 3 and 2-fold axes. I molecules have a unique 5-fold
        axis.
        """
        rot_present: dict[int, bool] = defaultdict(bool)
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        test_set = min(dist_el_sites.values(), key=len)
        coords = [s.coords for s in test_set]
        for c1, c2, c3 in itertools.combinations(coords, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                        rot_present[2] = self.is_valid_op(op)
                        if rot_present[2]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, 2))

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for r in (3, 4, 5):
                    if not rot_present[r]:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 360 / r)
                        rot_present[r] = self.is_valid_op(op)
                        if rot_present[r]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, r))
                            break
            if rot_present[2] and rot_present[3] and (rot_present[4] or rot_present[5]):
                break

    def get_symmetry_operations(self) -> Sequence[SymmOp]:
        """Get symmetry operations.

        Returns:
            list[SymmOp]: symmetry operations in Cartesian coord.
        """
        return generate_full_symmops(self.symmops, self.tol)

    def get_rotational_symmetry_number(self) -> int:
        """Get rotational symmetry number.

        Returns:
            int: Rotational symmetry number.
        """
        if self.sch_symbol == "D*h":
            # Special case. H2 for example has rotational symmetry number 2
            return 2

        """Get the rotational symmetry number."""
        symm_ops = self.get_symmetry_operations()
        symm_number = 0
        for symm in symm_ops:
            rot = symm.rotation_matrix
            if np.abs(np.linalg.det(rot) - 1) < 1e-4:
                symm_number += 1
        return symm_number

    def is_valid_op(self, symm_op: SymmOp) -> bool:
        """Check if a particular symmetry operation is a valid symmetry operation for a
        molecule, i.e., the operation maps all atoms to another equivalent atom.

        Args:
            symm_op (SymmOp): Symmetry operation to test.

        Returns:
            bool: True if SymmOp is valid for Molecule.
        """
        coords = self.centered_mol.cart_coords
        for site in self.centered_mol:
            coord = symm_op.operate(site.coords)
            ind = find_in_coord_list(coords, coord, self.tol)
            if len(ind) != 1 or self.centered_mol[ind[0]].species != site.species:
                return False
        return True
