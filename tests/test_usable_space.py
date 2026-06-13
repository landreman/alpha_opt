import numpy as np
import pytest
import alpha_opt.usable_space as usable_space
from alpha_opt.usable_space import measure_usable_space, measure_diversity


@pytest.mark.parametrize(
    "surface_type,vmec_input,which_nfp,self_intersection",
    [
        ("Garabedian01", "vacuum", "allNfp", True),
        ("Garabedian01", "vacuum", "allNfp", False),
        # ("Garabedian01", "finite beta", "allNfp"),
        ("Garabedian", "vacuum", "allNfp", True),
        ("Garabedian", "vacuum", "allNfp", False),
        ("Garabedian", "vacuum", "nfpAtLeast3", True),
        ("Garabedian", "vacuum", "nfpAtLeast3", False),
        ("Garabedian", "finite beta", "allNfp", True),
        ("Garabedian", "finite beta", "allNfp", False),
        ("Garabedian", "finite beta", "nfpAtLeast3", True),
        ("Garabedian", "finite beta", "nfpAtLeast3", False),
        ("PCA", "vacuum", "allNfp", True),
        ("PCA", "vacuum", "allNfp", False),
        ("PCA", "vacuum", "nfpAtLeast3", True),
        ("PCA", "vacuum", "nfpAtLeast3", False),
        ("PCA", "finite beta", "allNfp", True),
        ("PCA", "finite beta", "allNfp", False),
        ("PCA", "finite beta", "nfpAtLeast3", True),
        ("PCA", "finite beta", "nfpAtLeast3", False),
    ],
)
def test_usable_space(surface_type, vmec_input, which_nfp, self_intersection):
    print(f"Testing surface type: {surface_type}, VMEC input: {vmec_input}")
    n_trials, n_successes, n_good_iota, success_fraction, n_good_iota_fraction = (
        measure_usable_space(
            minutes=0.1,
            surface_type=surface_type,
            vmec_input=vmec_input,
            which_nfp=which_nfp,
            self_intersection=self_intersection,
            min_for_each_dof=0.1,
            n_pca_components=1,
            mpol=1,
            x_max=0.3,
            exponential_spectral_scaling=True,
        )
    )
    np.testing.assert_array_less(0, n_trials)
    np.testing.assert_array_less(0, n_successes)
    # np.testing.assert_array_less(success_fraction, 1.0)
    np.testing.assert_array_less(0.0, success_fraction)
    np.testing.assert_array_less(0, n_good_iota)
    # np.testing.assert_array_less(n_good_iota, n_trials)
    # np.testing.assert_array_less(n_good_iota_fraction, 1.0)
    np.testing.assert_array_less(0.0, n_good_iota_fraction)


def _install_fast_measure_usable_space_mocks(monkeypatch, garabedian_ndofs=7):
    calls = {
        "pca": [],
        "garabedian": [],
        "vmec": [],
        "get_objective": [],
    }

    class DummyVmec:
        def __init__(self, input_file, verbose=False):
            self.input_file = input_file
            self.verbose = verbose
            self._settings = {}
            self.indata = type("InData", (), {})()
            self.indata.nfp = None
            calls["vmec"].append(self)

        def set(self, key, value):
            self._settings[key] = value

        def get(self, key):
            return self._settings[key]

    def fake_surface_weighted_pca(*args, **kwargs):
        calls["pca"].append({"args": args, "kwargs": kwargs})
        # The dimension is the 4th positional argument in SurfaceWeightedPCA(...).
        dimension = int(args[3])
        surface = type("DummyWeightedPCASurface", (), {})()
        surface.x = np.zeros(dimension)
        return surface

    def fake_surface_garabedian_quantiles(**kwargs):
        calls["garabedian"].append({"kwargs": kwargs})
        surface = type("DummyGarabedianSurface", (), {})()
        surface.x = np.zeros(garabedian_ndofs)
        return surface

    def fake_get_objective(vmec, surface, x_scale, raw_objective, **kwargs):
        calls["get_objective"].append(
            {
                "vmec": vmec,
                "surface": surface,
                "x_scale": x_scale,
                "raw_objective": raw_objective,
                "kwargs": kwargs,
            }
        )

        # Returning fail_val ensures no success bookkeeping is attempted if loop runs.
        fail_val = kwargs.get("fail_val", 1e10)
        return lambda x: fail_val

    monkeypatch.setattr(usable_space, "mpi_available", False)
    monkeypatch.setattr(usable_space, "rank", 0)
    monkeypatch.setattr(usable_space, "size", 1)
    monkeypatch.setattr(usable_space, "Vmec", DummyVmec)
    monkeypatch.setattr(usable_space, "SurfaceWeightedPCA", fake_surface_weighted_pca)
    monkeypatch.setattr(usable_space, "SurfaceGarabedianQuantiles", fake_surface_garabedian_quantiles)
    monkeypatch.setattr(usable_space, "get_objective", fake_get_objective)
    return calls


def test_measure_usable_space_new_rejects_invalid_surface_type():
    with pytest.raises(ValueError, match="surface_type"):
        usable_space.measure_usable_space(surface_type="bad", minutes=0)


def test_measure_usable_space_new_rejects_invalid_which_nfp():
    with pytest.raises(ValueError, match="which_nfp"):
        usable_space.measure_usable_space(which_nfp="bad", minutes=0)


@pytest.mark.parametrize(
    "surface_type,which_nfp,expected_suffix",
    [
        ("PCA", "allNfp", "20260401-01_prepare_weighted_data_allNfp_PCA.h5"),
        ("PCA", "nfpAtLeast3", "20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5"),
        ("Garabedian", "allNfp", "20260401-01_prepare_weighted_data_allNfp_Garabedian.h5"),
        ("Garabedian", "nfpAtLeast3", "20260402-01_prepare_weighted_data_nfpAtLeast3_Garabedian.h5"),
    ],
)
def test_measure_usable_space_new_selects_expected_data_file(
    monkeypatch, surface_type, which_nfp, expected_suffix
):
    calls = _install_fast_measure_usable_space_mocks(monkeypatch)

    usable_space.measure_usable_space(
        surface_type=surface_type,
        which_nfp=which_nfp,
        n_pca_components=5,
        mpol=4,
        minutes=0,
    )

    if surface_type == "PCA":
        assert len(calls["pca"]) == 1
        filename = calls["pca"][0]["kwargs"]["filename"]
    else:
        assert len(calls["garabedian"]) == 1
        filename = calls["garabedian"][0]["kwargs"]["filename"]

    assert filename.endswith(expected_suffix)


@pytest.mark.parametrize(
    "vmec_input,expected_suffix",
    [
        ("vacuum", "input.vmec"),
        ("finite beta", "input.finite_beta"),
    ],
)
def test_measure_usable_space_new_maps_named_vmec_inputs(monkeypatch, vmec_input, expected_suffix):
    calls = _install_fast_measure_usable_space_mocks(monkeypatch)

    usable_space.measure_usable_space(vmec_input=vmec_input, minutes=0)

    assert len(calls["vmec"]) == 1
    assert calls["vmec"][0].input_file.endswith(expected_suffix)


def test_measure_usable_space_new_keeps_custom_vmec_input_path(monkeypatch):
    calls = _install_fast_measure_usable_space_mocks(monkeypatch)
    custom_input = "/tmp/custom_input.vmec"

    usable_space.measure_usable_space(vmec_input=custom_input, minutes=0)

    assert len(calls["vmec"]) == 1
    assert calls["vmec"][0].input_file == custom_input


def test_measure_usable_space_new_wires_objective_and_phiedge(monkeypatch):
    garabedian_ndofs = 11
    calls = _install_fast_measure_usable_space_mocks(
        monkeypatch, garabedian_ndofs=garabedian_ndofs
    )

    max_B_target = 13.7
    max_B_iterations = 3
    mpol = 5
    usable_space.measure_usable_space(
        surface_type="Garabedian",
        which_nfp="allNfp",
        mpol=mpol,
        max_B_target=max_B_target,
        max_B_iterations=max_B_iterations,
        minutes=0,
    )

    assert len(calls["garabedian"]) == 1
    gar_kwargs = calls["garabedian"][0]["kwargs"]
    assert gar_kwargs["mpol"] == mpol
    assert gar_kwargs["ntor"] == mpol

    assert len(calls["get_objective"]) == 1
    objective_call = calls["get_objective"][0]
    np.testing.assert_equal(len(objective_call["x_scale"]), garabedian_ndofs)

    expected_phiedge = (
        2 * np.pi * (max_B_target / np.sqrt(2)) * usable_space.ARIES_CS_MINOR_RADIUS**2
    )
    np.testing.assert_allclose(objective_call["kwargs"]["phiedge"], expected_phiedge)
    np.testing.assert_equal(objective_call["kwargs"]["max_B"], max_B_target)
    np.testing.assert_equal(objective_call["kwargs"]["max_B_iterations"], max_B_iterations)

    assert len(calls["vmec"]) == 1
    np.testing.assert_allclose(calls["vmec"][0]._settings["phiedge"], expected_phiedge)

def test_diversity():
    # If you run for longer time, the results shouldn't change much:
    surface_type="Garabedian01"
    x_max = 0.01
    minutes1 = 0.03
    mpol = 3
    n_trials1, sum_abs_distances1, sum_distances_squared1, mean_abs_distances1, mean_distances_squared1 = measure_diversity(
        surface_type=surface_type,
        minutes=minutes1,
        x_max=x_max,
        mpol=mpol,
    )
    n_trials2, sum_abs_distances2, sum_distances_squared2, mean_abs_distances2, mean_distances_squared2 = measure_diversity(
        surface_type=surface_type,
        minutes=minutes1 * 2,
        x_max=x_max,
        mpol=mpol,
    )
    np.testing.assert_array_less(n_trials1, n_trials2)
    np.testing.assert_array_less(sum_abs_distances1, sum_abs_distances2)
    np.testing.assert_array_less(sum_distances_squared1, sum_distances_squared2)
    np.testing.assert_allclose(mean_abs_distances1, mean_abs_distances2, rtol=0.02)
    np.testing.assert_allclose(mean_distances_squared1, mean_distances_squared2, rtol=0.03)

    # If you increase x_max, the distances should increase:
    n_trials3, sum_abs_distances3, sum_distances_squared3, mean_abs_distances3, mean_distances_squared3 = measure_diversity(
        surface_type=surface_type,
        minutes=minutes1,
        x_max=x_max * 10,
        mpol=mpol,
    )
    np.testing.assert_array_less(mean_abs_distances1 * 8, mean_abs_distances3)
    np.testing.assert_array_less(mean_distances_squared1 * 80, mean_distances_squared3)
