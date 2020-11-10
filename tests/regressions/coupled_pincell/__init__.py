import pathlib
import shutil
import warnings

import numpy
import numpy.testing
import pytest

h5py = pytest.importorskip("h5py")

from tests.regressions import CompareBase  # noqa: E402


class HdfResultCompare(CompareBase):
    targetFile = "hydep-results.h5"
    expectsVersion = (0, 1)

    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent)

    def main(self, integrator):
        integrator.integrate()
        res = pathlib.Path(self.targetFile).resolve()
        assert res.is_file()

        self._checkConsistency(integrator, res)

        return super().main(res)

    def _checkConsistency(self, problem, res):
        with h5py.File(res, mode="r") as hf:
            materialIDs = hf["materials/ids"]
            names = hf["materials/names"]
            for ix, mat in enumerate(
                problem.model.root.findBurnableMaterials()
            ):
                assert mat.id == materialIDs[ix]
                assert mat.name == names[ix].decode()
            assert hf.attrs["burnableMaterials"] == len(names)

            assert hf.attrs["isotopes"] == len(problem.dep.chain)
            zais = hf["isotopes/zais"]
            names = hf["isotopes/names"]
            for ix, iso in enumerate(problem.dep.chain):
                assert iso.zai == zais[ix]
                assert iso.name == names[ix].decode()

            assert hf.attrs["coarseSteps"] == len(problem.dep.timesteps) + 1
            assert hf.attrs["totalSteps"] == sum(problem.dep.substeps) + 1

    def update(self, testF):
        shutil.move(str(testF), str(self.datadir / "results-reference.h5"))

    def compare(self, testF):
        reference = self.datadir / "results-reference.h5"

        with h5py.File(testF, mode="r") as test, h5py.File(
            reference, mode="r"
        ) as ref:
            version = test.attrs.get("fileVersion")
            if version is None:
                raise ValueError("HDF storage version not written")
            elif tuple(version[:]) != self.expectsVersion:
                raise ValueError(
                    "HDF storage have been updated beyond this test. Found "
                    f"{version[:]}, expected {self.expectsVersion}"
                )

            try:
                self._compareK(test, ref)
                self._compareComps(test, ref)
            except AssertionError:
                shutil.move(str(testF), str(self.datadir / "results-fail.h5"))
                raise

    def _compareK(self, test, ref):
        actualDS = test["multiplicationFactor"]
        refK = ref["multiplicationFactor"]

        if actualDS[:, 0] == pytest.approx(refK[:, 0]):
            return

        refnan = numpy.isnan(refK).any(axis=1)
        actualnan = numpy.isnan(actualDS).any(axis=1)

        assert (refnan == actualnan).all(), (refnan, actualnan)

        # Check statistics
        diff = numpy.fabs(actualDS[~refnan, 0] - refK[~refnan, 0])

        unc = numpy.sqrt(
            numpy.square(actualDS[~refnan, 1]) + numpy.square(refK[~refnan, 1])
        )

        assert (diff <= unc).all(), (diff, unc)

    def _compareComps(self, test, ref):
        actual = test["compositions"]
        reference = ref["compositions"]

        # Check BOL, EOL, and then internals
        # Internals will be skipped with warning if the number
        # of intermediate steps (substeps) differs
        assert actual[0] == pytest.approx(reference[0])

        assert actual[-1] == pytest.approx(reference[-1])

        if len(actual) != len(reference):
            raise pytest.skip(
                f"Number of steps differs: Reference: {len(reference)} "
                f"Test: {len(actual)}. Contents will not be tested",
                RuntimeWarning,
            )
        assert actual[1:-1] == pytest.approx(reference[1:-1])
