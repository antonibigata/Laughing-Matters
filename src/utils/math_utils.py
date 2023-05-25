import numpy as np
from scipy import spatial
from scipy.spatial._qhull import QhullError


def furthest_away_pts(pts):
    """
    Find the two furthest away points in a set of points
    """

    # two points which are fruthest apart will occur as vertices of the convex hull
    try:
        candidates = pts[spatial.ConvexHull(pts).vertices]
    except QhullError:
        return None, None

    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return i, j


if __name__ == "__main__":
    pts = np.random.rand(100, 3)
    i, j = furthest_away_pts(pts)
    print(i, j)
