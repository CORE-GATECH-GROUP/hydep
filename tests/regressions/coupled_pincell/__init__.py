import pathlib
import shutil

import numpy
import numpy.testing
import pytest

h5py = pytest.importorskip("h5py")

from tests.regressions import CompareBase


class HdfResultCompare(CompareBase):
    targetFile = "hydep-results.h5"
    expectsVersion = (0, 1)

    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent)

    def main(self, problem):
        problem.solve()
        res = pathlib.Path(self.targetFile).resolve()
        assert res.is_file(), res

        self._checkConsistency(problem, res)

        return super().main(res)

    def _checkConsistency(self, problem, res):
        with h5py.File(res, mode="r") as hf:
            materialIDs = hf["materials/ids"]
            names = hf["materials/names"]
            for ix, mat in enumerate(
                problem.model.root.findBurnableMaterials()
            ):
                assert mat.id == materialIDs[ix], (mat.id, materialIDs[ix])
                assert mat.name == names[ix].decode(), (
                    mat.name,
                    names[ix].decode(),
                )
            assert hf.attrs["burnableMaterials"] == len(names)

            assert hf.attrs["isotopes"] == len(problem.dep.chain)
            zais = hf["isotopes/zais"]
            names = hf["isotopes/names"]
            for ix, iso in enumerate(problem.dep.chain):
                assert iso.zai == zais[ix], (iso, zais[ix])
                assert iso.name == names[ix].decode(), (
                    iso,
                    names[ix].decode(),
                )

            assert hf.attrs["coarseSteps"] == len(problem.dep.timesteps) + 1
            assert hf.attrs["totalSteps"] == sum(problem.dep.substeps) + 1

    def update(self, testF):
        shutil.move(str(testF), str(self.datadir / "results-reference.h5"))

    def compare(self, testF):
        reference = self.datadir / "results-reference.h5"
        fails = []

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

            if not self._compareK(test, ref):
                fails.append("keff")

        if fails:
            shutil.move(str(testF), str(self.datadir / "results-fail.h5"))
        return fails

    def _compareK(self, test, ref):
        actualDS = test.get("multiplicationFactor")
        if actualDS is None:
            return False

        refK = ref["multiplicationFactor"]

        if actualDS[:, 0] == pytest.approx(refK[:, 0]):
            return True

        refnan = numpy.isnan(refK).any(axis=1)
        actualnan = numpy.isnan(actualDS).any(axis=1)

        if not numpy.equal(refnan, actualnan).any():
            return False

        # Check statistics
        diff = numpy.fabs(actualDS[~refnan, 0] - refK[~refnan, 0])

        unc = numpy.sqrt(
            numpy.square(actualDS[~refnan, 1]) + numpy.square(refK[~refnan, 1])
        )

        # Check against non-nans
        if (diff > unc).any():
            return False

        return True
