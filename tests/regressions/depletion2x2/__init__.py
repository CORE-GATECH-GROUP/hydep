import warnings

import numpy
import pytest

from tests.regressions import CompareBase


class DepletionComparator(CompareBase):
    floatFormat = "% .12E"
    intFormat = "%7d"

    @staticmethod
    def _bundleToMatrix(compBundle):
        # Matrix with ZAI as first column, densities in remaining columns
        mtx = numpy.empty(
            (len(compBundle.isotopes), 1 + len(compBundle.densities)), order="F"
        )
        for rx, iso in enumerate(compBundle.isotopes):
            mtx[rx, 0] = iso.zai

        for cx, dens in enumerate(compBundle.densities, start=1):
            mtx[:, cx] = dens
        return mtx

    def writeCompBundle(self, compBundle, status):
        mtx = self._bundleToMatrix(compBundle)
        fmt = " ".join(
            [self.intFormat] + [self.floatFormat] * len(compBundle.densities)
        )

        with self.getPathFor("concentrations", status).open("w") as stream:
            numpy.savetxt(stream, mtx, fmt=fmt)

    def main(self, compBundle):
        super().main(compBundle)

    def update(self, compBundle):
        return self.writeCompBundle(compBundle, "reference")

    def compare(self, compBundle):
        asmtx = self._bundleToMatrix(compBundle)
        reference = numpy.loadtxt(self.getPathFor("concentrations", "reference"))

        # Get common indexes
        testzai = tuple(asmtx[:, 0].astype(int))
        refzai = tuple(reference[:, 0].astype(int))

        common = set(testzai).intersection(refzai)

        assert common, "No shared isotopes"

        if len(common) != len(testzai) != len(refzai):
            testmissing = []
            for zai in set(testzai).difference(refzai):
                testmissing.append(asmtx[testzai.index(zai), 1:])

            testsum = numpy.sum(testmissing, axis=0)

            refmissing = []
            for zai in set(refzai).difference(testzai):
                refmissing.append(reference[refzai.index(zai), 1:])

            refsum = numpy.sum(refmissing, axis=0)

            if testsum != pytest.approx(0) or refsum != pytest.approx(0):
                testmsg = (
                    f"{len(testmissing)} isotopes, contributing {testsum} [#/b-cm]"
                )
                refmsg = f"{len(refmissing)} isotopes, contributing {refsum} [#/b-cm]"
                warnings.warn(
                    f"Some isotopes found only in test and/or reference data:\n"
                    f"\tTest: {testmsg}\n\tReference: {refmsg}"
                )

        fails = []
        for z in sorted(common):
            testrow = testzai.index(z)
            refrow = refzai.index(z)
            if asmtx[testrow, 1:] == pytest.approx(reference[refrow, 1:]):
                continue
            fails.append((z, testrow, refrow))

        if not fails:
            return []

        fmt = " ".join(
            [self.intFormat] + [self.floatFormat] * len(compBundle.densities)
        )
        with self.getPathFor("concentrations", "fail").open("w") as stream:
            numpy.savetxt(stream, asmtx, fmt=fmt)

        # Compute absolute relative error against reference data
        rdiff = numpy.empty((len(fails), 1 + len(compBundle.densities)), order="f")
        indexes = numpy.array(fails, dtype=int, order="f")
        rdiff[:, 0] = indexes[:, 0]

        referror = reference[indexes[:, 2], 1:]
        rdiff[:, 1:] = numpy.fabs(asmtx[indexes[:, 1], 1:] - referror)
        nzr, nzc = referror.nonzero()
        rdiff[nzr, nzc + 1] /= referror[nzr, nzc]

        # Sort by highest average absolute relative error
        sortix = rdiff.mean(axis=1).argsort()[::-1]

        with self.getPathFor("relative_errors", "fail").open("w") as stream:
            numpy.savetxt(
                stream,
                rdiff[sortix],
                fmt=fmt,
                header="Sorted by highest average absolute relative difference",
            )

        # Return list of highest failures
        return [fails[x][0] for x in sortix]
