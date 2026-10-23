"""
test_moo_utils.py -- Unit tests for the shared toolkit.
Tests: dominance (car example), fast non-dominated sort, crowding distance,
2-D hypervolume, Monte-Carlo hypervolume, IGD, spacing, das_dennis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from moo_utils import (dominates, constrained_dominates, fast_nondominated_sort,
                       crowding_distance, normalize, hypervolume_2d, hypervolume_mc,
                       hypervolume, igd, spacing, das_dennis)


def test_dominance_car_example():
    """Lecture car example: cheaper AND faster wins."""
    car_A = np.array([20000, 6.0])
    car_B = np.array([25000, 7.0])
    car_C = np.array([20000, 7.0])
    assert dominates(car_A, car_B), "A should dominate B"
    assert dominates(car_A, car_C), "A should dominate C"
    assert not dominates(car_C, car_A), "C should not dominate A"
    print("PASS: dominance (car example)")


def test_constrained_dominance_deb():
    """Deb's 3 rules: feasible > infeasible; smaller violation among infeasible; Pareto among feasible."""
    # Rule 1: feasible beats infeasible
    assert constrained_dominates(np.array([10,10]), 0.0, np.array([5,5]), 1.0)
    assert not constrained_dominates(np.array([5,5]), 1.0, np.array([10,10]), 0.0)
    # Rule 2: both infeasible -> smaller violation wins
    assert constrained_dominates(np.array([0,0]), 0.5, np.array([0,0]), 1.0)
    # Rule 3: both feasible -> Pareto dominance
    assert constrained_dominates(np.array([1,1]), 0.0, np.array([2,2]), 0.0)
    assert not constrained_dominates(np.array([2,2]), 0.0, np.array([1,1]), 0.0)
    print("PASS: constrained_dominates (Deb's 3 rules)")


def test_fast_nondominated_sort():
    """Tiny hand-checkable set. Front 0 = {0,1,2}."""
    F = np.array([[1,4],[2,2],[3,1],[4,4],[2,3],[3,3]], dtype=float)
    fronts, rank = fast_nondominated_sort(F)
    assert set(fronts[0]) == {0,1,2}, f"Expected front 0={{0,1,2}}, got {set(fronts[0])}"
    print(f"PASS: fast_nondominated_sort: fronts={fronts}")


def test_crowding_distance():
    """Boundary points must have infinite crowding distance."""
    F = np.array([[1,4],[2,2],[3,1],[4,4],[2,3],[3,3]], dtype=float)
    fronts, _ = fast_nondominated_sort(F)
    cd = crowding_distance(F, fronts[0])
    assert cd[0] == np.inf and cd[2] == np.inf, "Boundary points must get inf crowding distance"
    print(f"PASS: crowding_distance: {cd}")


def test_hypervolume_2d():
    """Cross-check against known value."""
    front2 = np.array([[1,4],[2,2],[4,1]], dtype=float)
    ref = np.array([5,5], dtype=float)
    hv = hypervolume_2d(front2, ref)
    assert abs(hv - 11.0) < 1e-9, f"HV expected 11.0, got {hv}"
    print(f"PASS: hypervolume_2d: HV={hv}")


def test_hypervolume_mc_3d():
    """Monte-Carlo HV for a single 3D point should be exactly 1."""
    hv3 = hypervolume_mc(np.array([[1,1,1]], float), np.array([2,2,2], float),
                          n_samples=500000, seed=1)
    assert abs(hv3 - 1.0) < 0.05, f"MC HV expected ~1.0, got {hv3}"
    print(f"PASS: hypervolume_mc (3D single point): HV~{hv3:.4f}")


def test_igd_self():
    """IGD = 0 when obtained == true front."""
    front2 = np.array([[1,4],[2,2],[4,1]], dtype=float)
    assert igd(front2, front2) < 1e-9, "IGD of front with itself should be 0"
    print("PASS: IGD self-test")


def test_spacing_even():
    """Spacing = 0 for perfectly evenly spaced points."""
    even = np.array([[0,2],[1,1],[2,0]], dtype=float)
    s = spacing(even)
    assert s < 1e-9, f"Spacing of evenly spaced points should be 0, got {s}"
    print(f"PASS: spacing even: {s}")


def test_das_dennis():
    """For M=3, p=2: expect 6 reference points summing to 1."""
    pts = das_dennis(3, 2)
    assert pts.shape == (6, 3), f"Expected shape (6,3), got {pts.shape}"
    assert np.allclose(pts.sum(axis=1), 1.0), "All reference points should sum to 1"
    print(f"PASS: das_dennis (M=3, p=2): {pts.shape[0]} points")


if __name__ == '__main__':
    test_dominance_car_example()
    test_constrained_dominance_deb()
    test_fast_nondominated_sort()
    test_crowding_distance()
    test_hypervolume_2d()
    test_hypervolume_mc_3d()
    test_igd_self()
    test_spacing_even()
    test_das_dennis()
    print("\nAll moo_utils tests PASSED.")
