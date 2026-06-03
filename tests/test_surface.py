import os
import numpy as np
from vmecpp.simsopt_compat import Vmec
from simsopt.geo import SurfaceGarabedian
from alpha_opt import init_optimizable_surface, SurfaceGarabedianQuantiles, SurfaceGarabedian01, DATA_DIR


def test_init_optimizable_surface():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5
    for mn_max in [1, 2, 3]:
        surface, dim_x, x_scale, x0 = init_optimizable_surface(
            mn_max, mn_max, nfp, major_radius, minor_radius
        )


def test_surface_Garabedian_quantiles():
    nfp = 3
    major_radius = 3.3
    minor_radius = 0.2

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=1,
        ntor=1,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    np.testing.assert_equal(len(surface.x), 7)
    np.testing.assert_allclose(
        surface.to_RZFourier().major_radius(), major_radius, rtol=0.007
    )
    np.testing.assert_allclose(
        surface.to_RZFourier().minor_radius(), minor_radius, rtol=0.12
    )

    surface.x = np.ones_like(surface.x) * 0.6
    np.testing.assert_equal(len(surface.x), 7)
    np.testing.assert_allclose(
        surface.to_RZFourier().major_radius(), major_radius, rtol=0.011
    )
    np.testing.assert_allclose(
        surface.to_RZFourier().minor_radius(), minor_radius, rtol=0.1
    )

    surface2 = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    np.testing.assert_equal(len(surface2.x), 5 * 7 - 2)


def test_surface_Garabedian_quantiles_exact_radii():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
        exact_radii=True,
    )
    surface2 = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
        exact_radii=False,
    )

    # Perturb controls so exact_radii enforcement is exercised on recompute.
    surface.x = np.ones_like(surface.x) * 0.6
    surface2.x = surface.x
    rz_surface = surface.to_RZFourier()
    rz_surface2 = surface2.to_RZFourier()
    print("Final major radius:", rz_surface.major_radius(), "minor radius:", rz_surface.minor_radius())

    np.testing.assert_allclose(
        rz_surface.major_radius(), major_radius, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_allclose(
        rz_surface.minor_radius(), minor_radius, atol=1e-10, rtol=1e-12
    )
    # Surfaces should be identical up to the overall scale and the major radius
    np.testing.assert_allclose(
        rz_surface.x[1:] * rz_surface2.minor_radius() / rz_surface.minor_radius(),
        rz_surface2.x[1:],
    )


def test_surface_Garabedian_quantiles_regression():
    """Compare to 20260306-02_weightedQuantile_on_hdf5_Garabedian_interactive.py"""
    nfp = 3
    minor_radius = 0.2
    major_radius = 10.0 * minor_radius

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=1,
        ntor=1,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    reference_x = np.array([-9.58024190e-02, 1.00000000e+00, 1.04883385e-01, 4.40607174e-02,
                            1.00000000e+01, 7.56137942e-01, -7.48472419e-04, -8.59236577e-02,
                            -4.71235395e-01])

    np.testing.assert_allclose(surface.surface_garabedian.x, reference_x * minor_radius, rtol=1e-8)


def test_surface_Garabedian_quantiles_with_vmec():
    vmec = Vmec(os.path.join(DATA_DIR, "input.vmec"))

    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5

    surface = SurfaceGarabedianQuantiles(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    vmec.boundary = surface
    vmec.indata.nfp = (
        nfp  # Vmec++ does not automatically get nfp from the boundary surface!
    )

    vmec.run()
    np.testing.assert_equal(vmec.wout.nfp, nfp)
    np.testing.assert_allclose(vmec.wout.Rmajor_p, major_radius, rtol=0.02)
    np.testing.assert_allclose(vmec.wout.Aminor_p, minor_radius, rtol=0.1)
    print("iota:", list(float(x) for x in vmec.wout.iotaf))
    np.testing.assert_allclose(vmec.wout.iotaf[0], 0.7235724832954784)
    np.testing.assert_allclose(vmec.wout.iotaf[-1], 0.7833059615881733)

def test_surface_Garabedian_01_dof_ranges():
    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5
    x_max = 0.08

    for mpol in [1, 2]:
        for ntor in [0, 3]:
            surface = SurfaceGarabedian01(
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                major_radius=major_radius,
                minor_radius=minor_radius,
                exponential_spectral_scaling=False,
                exact_radii=False,
                x_max=x_max,
            )
            # If dofs are 1, the Delta_m,n should be x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.ones_like(surface.x)
            surface2 = surface.to_RZFourier()
            # Since exact_radii=False, the major and minor radius are only
            # matched approximately.
            np.testing.assert_allclose(surface2.major_radius(), major_radius, rtol=0.05)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius, rtol=0.01)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, np.ones_like(surface3.x) * minor_radius * x_max)
            factor1 = surface2.minor_radius() / minor_radius

            # If dofs are 0, the Delta_m,n should be -x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.zeros_like(surface.x)
            surface2 = surface.to_RZFourier()
            np.testing.assert_allclose(surface2.major_radius(), major_radius, rtol=0.02)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius, rtol=0.01)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, -np.ones_like(surface3.x) * minor_radius * x_max)
            factor0 = surface2.minor_radius() / minor_radius

            # Now try with exact_radii=True.
            surface = SurfaceGarabedian01(
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                major_radius=major_radius,
                minor_radius=minor_radius,
                exponential_spectral_scaling=False,
                exact_radii=True,
                x_max=x_max,
            )
            # If dofs are 1, the Delta_m,n should be x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.ones_like(surface.x)
            surface2 = surface.to_RZFourier()
            # Since exact_radii=True, the major and minor radius should be exactly correct.
            np.testing.assert_allclose(surface2.major_radius(), major_radius)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, np.ones_like(surface3.x) * minor_radius * x_max / factor1)

            # If dofs are 0, the Delta_m,n should be -x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.zeros_like(surface.x)
            surface2 = surface.to_RZFourier()
            np.testing.assert_allclose(surface2.major_radius(), major_radius)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, -np.ones_like(surface3.x) * minor_radius * x_max / factor0)

            # Now try with exponential_spectral_scaling=True.
            surface = SurfaceGarabedian01(
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                major_radius=major_radius,
                minor_radius=minor_radius,
                exponential_spectral_scaling=True,
                exact_radii=False,
                x_max=x_max,
            )
            np.testing.assert_array_less(surface.x_scale, 1.0 + 1e-8)
            np.testing.assert_array_less(0.0, surface.x_scale)
            np.testing.assert_allclose(max(surface.x_scale), 1.0)
            # If dofs are 1, the Delta_m,n should be x_max * minor_radius * x_scale
            # (except for the major and minor radius)
            surface.x = np.ones_like(surface.x)
            surface2 = surface.to_RZFourier()
            # Since exact_radii=False, the major and minor radius are only
            # matched approximately.
            np.testing.assert_allclose(surface2.major_radius(), major_radius, rtol=0.05)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius, rtol=0.01)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, np.ones_like(surface3.x) * minor_radius * x_max * surface.x_scale)
            factor1 = surface2.minor_radius() / minor_radius

            # If dofs are 0, the Delta_m,n should be -x_max * minor_radius * x_scale
            # (except for the major and minor radius)
            surface.x = np.zeros_like(surface.x)
            surface2 = surface.to_RZFourier()
            np.testing.assert_allclose(surface2.major_radius(), major_radius, rtol=0.02)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius, rtol=0.01)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, -np.ones_like(surface3.x) * minor_radius * x_max * surface.x_scale)
            factor0 = surface2.minor_radius() / minor_radius

            # Now try with exact_radii=True.
            surface = SurfaceGarabedian01(
                nfp=nfp,
                mpol=mpol,
                ntor=ntor,
                major_radius=major_radius,
                minor_radius=minor_radius,
                exponential_spectral_scaling=True,
                exact_radii=True,
                x_max=x_max,
            )
            np.testing.assert_array_less(surface.x_scale, 1.0 + 1e-8)
            np.testing.assert_array_less(0.0, surface.x_scale)
            np.testing.assert_allclose(max(surface.x_scale), 1.0)
            # If dofs are 1, the Delta_m,n should be x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.ones_like(surface.x)
            surface2 = surface.to_RZFourier()
            # Since exact_radii=True, the major and minor radius should be exactly correct.
            np.testing.assert_allclose(surface2.major_radius(), major_radius)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, np.ones_like(surface3.x) * minor_radius * x_max * surface.x_scale / factor1)

            # If dofs are 0, the Delta_m,n should be -x_max * minor_radius
            # (except for the major and minor radius)
            surface.x = np.zeros_like(surface.x)
            surface2 = surface.to_RZFourier()
            np.testing.assert_allclose(surface2.major_radius(), major_radius)
            np.testing.assert_allclose(surface2.minor_radius(), minor_radius)
            surface3 = SurfaceGarabedian.from_RZFourier(surface2)
            surface3.fix("Delta(0,0)")  # Minor radius
            surface3.fix("Delta(1,0)")  # Major radius
            np.testing.assert_allclose(surface3.x, -np.ones_like(surface3.x) * minor_radius * x_max * surface.x_scale / factor0)

def test_surface_Garabedian_01_with_vmec():
    vmec = Vmec(os.path.join(DATA_DIR, "input.vmec"))

    nfp = 3
    major_radius = 2.3
    minor_radius = 0.5

    surface = SurfaceGarabedian01(
        nfp=nfp,
        mpol=2,
        ntor=3,
        major_radius=major_radius,
        minor_radius=minor_radius,
        exact_radii=True,
        exponential_spectral_scaling=True,
        x_max=0.2,
    )
    vmec.boundary = surface
    vmec.indata.nfp = (
        nfp  # Vmec++ does not automatically get nfp from the boundary surface!
    )
    surface.x = [0.2919483 , 0.01458784, 0.33959851, 0.67075689, 0.85201523,
       0.40368797, 0.31449363, 0.70907725, 0.56704031, 0.21109265,
       0.55388917, 0.07341619, 0.15555743, 0.7092611 , 0.60333879,
       0.42050352, 0.58377448, 0.28925482, 0.98806487, 0.62794344,
       0.12269605, 0.96256941, 0.60453468, 0.98045459, 0.64625453,
       0.0469208 , 0.51604649, 0.39349023, 0.7395622 , 0.69558463,
       0.6184542 , 0.31259228, 0.53705241]

    vmec.run()
    np.testing.assert_equal(vmec.wout.nfp, nfp)
    np.testing.assert_allclose(vmec.wout.Rmajor_p, major_radius)
    np.testing.assert_allclose(vmec.wout.Aminor_p, minor_radius)
    print("iota:", list(float(x) for x in vmec.wout.iotaf))
    np.testing.assert_allclose(vmec.wout.iotaf[0], 0.013200207160874146)
    np.testing.assert_allclose(vmec.wout.iotaf[-1], 0.020488826401016888)