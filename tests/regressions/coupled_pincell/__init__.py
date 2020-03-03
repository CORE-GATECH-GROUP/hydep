import pathlib
import shutil

import numpy
import numpy.testing
import pytest

h5py = pytest.importorskip("h5py")

from tests.regressions import CompareBase


class HdfResultCompare(CompareBase):
    targetFile = "hydep-results.h5"
    expectsVersion = (0, 0)

    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent)

    def main(self, problem):
        problem.solve()
        res = pathlib.Path(self.targetFile).resolve()
        assert res.is_file(), res

        return super().main(res)

    def update(self, testF):
        shutil.move(str(testF), str(self.datadir / "results-reference.h5"))

    def compare(self, testF):
        reference = self.datadir / "results-reference.h5"
        fails = []

        with h5py.File(testF, mode="r") as test, h5py.File(reference, mode="r") as ref:
            version = test.attrs.get("file version")
            if version is None or tuple(version[:]) != self.expectsVersion:
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
        actualDS = test.get("multiplication factor")
        if actualDS is None:
            return False
        refK = ref["multiplication factor"]

        if actualDS[:, 0] == pytest.approx(refK[:, 0]):
            return True

        # Check statistics
        diff = numpy.fabs(actualDS[:, 0] - refK[:, 0])

        unc = numpy.sqrt(numpy.square(actualDS[:, 1]) + numpy.square(refK[:, 1]))

        if (diff > unc).any():
            return False

        return True
