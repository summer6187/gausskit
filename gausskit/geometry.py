import numpy as np

def angle(v1, v2):
    """
    Compute the angle between two vectors in radians.

    The function returns the unsigned angle (0 ≤ θ ≤ π) between two
    vectors `v1` and `v2`, computed using the dot product formula:

        θ = arccos( (v1 · v2) / (|v1| |v2|) )

    Parameters
    ----------
    v1 : array_like
        First input vector (1D NumPy array of shape (3,) or similar).
    v2 : array_like
        Second input vector (same shape as `v1`).

    Returns
    -------
    float
        Angle between the two vectors, in radians.

    Notes
    -----
    - The result is clipped to the valid domain of arccos ([-1, 1])
      to avoid numerical errors from floating-point rounding.
    - To obtain the angle in degrees, wrap the result with
      `np.degrees(angle(v1, v2))`.

    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([0, 1, 0])
    >>> np.degrees(angle(v1, v2))
    90.0
    """
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))

def dihedral(p):
    """
    Compute the dihedral (torsion) angle defined by four points.

    Given four atomic coordinates `p0`, `p1`, `p2`, and `p3`,
    this function returns the signed dihedral angle (in radians)
    between the plane formed by points (p0, p1, p2)
    and the plane formed by (p1, p2, p3).

    The formula used is based on an older version of the
    Wikipedia article on "Dihedral angle":
    https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors

    This implementation uses three cross products and one square root,
    and follows the algorithm described here:
    https://stackoverflow.com/a/34245697

    Parameters
    ----------
    p : (4, 3) array_like
        Array of four points, where each point is a 3D coordinate.
        Typically, `p[i]` corresponds to the Cartesian coordinates
        of atom i.

    Returns
    -------
    float
        Dihedral (torsion) angle in radians. The angle is signed:
        a positive value corresponds to a right-hand (clockwise)
        rotation from the first plane to the second.

    Notes
    -----
    - To convert the result to degrees, use `np.degrees(dihedral(p))`.
    - The angle is computed as `atan2(y, x)` for better numerical stability.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0],
    ...     [1.0, 1.0, 1.0]
    ... ])
    >>> np.degrees(dihedral(p))
    90.0
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.arctan2(y, x)
