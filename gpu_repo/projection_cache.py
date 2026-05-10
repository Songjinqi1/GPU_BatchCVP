import numpy as np


def integer_matrix_to_numpy(B):
    """
    fpylll IntegerMatrix -> np.ndarray[float64]
    """
    nrows, ncols = B.nrows, B.ncols
    out = np.zeros((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            out[i, j] = float(B[i, j])
    return out


def build_projected_block(basis_np: np.ndarray, ell: int):
    """
    basis_np: row-basis form, shape (d, d)

    We convert to column-basis form B_col = basis_np.T and compute:
      B_col = Q R

    Then keep the last ell coordinates:
      Q2T = Q[:, -ell:].T
      R22 = R[-ell:, -ell:]
    """
    B_col = basis_np.T
    Q, R = np.linalg.qr(B_col)
    Q2T = Q[:, -ell:].T
    R22 = R[-ell:, -ell:]
    return Q2T, R22


class ProjectionCache:
    """
    Cache for per-instance projected blocks.
    Typical key:
      (lat_index, chosen_delta)

    Each item may contain:
      {
        "basis_np": ...,
        "Q2T": ...,
        "R22": ...,
        "ell": ...,
        "_gpu_Q2T": ...,
        "_gpu_R22": ...,
      }
    """
    def __init__(self):
        self._cache = {}

    def has(self, key):
        return key in self._cache

    def put(self, key, value):
        self._cache[key] = value

    def get(self, key):
        return self._cache[key]

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def ensure_gpu_arrays(self, key, cp):
        """
        Lazily materialize and cache CuPy arrays for a cache entry.
        """
        item = self._cache[key]

        if "_gpu_Q2T" not in item:
            item["_gpu_Q2T"] = cp.asarray(item["Q2T"], dtype=cp.float64)
        if "_gpu_R22" not in item:
            item["_gpu_R22"] = cp.asarray(item["R22"], dtype=cp.float64)

        return item["_gpu_Q2T"], item["_gpu_R22"]

    def describe(self):
        out = {}
        for k, v in self._cache.items():
            out[str(k)] = {
                "ell": int(v.get("ell", -1)),
                "has_gpu_Q2T": "_gpu_Q2T" in v,
                "has_gpu_R22": "_gpu_R22" in v,
            }
        return out