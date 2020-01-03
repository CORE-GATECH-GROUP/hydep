import filecmp
import difflib


def showStringDiff(reference, written, fromfile="reference", tofile="actual"):
    diff = difflib.unified_diff(
        reference.splitlines(keepends=True),
        written.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    print("".join(diff))


def strcompare(reference, written):
    if reference == written:
        return True
    showStringDiff(reference, written)
    return False


def filecompare(rpath, apath, failpath):
    if filecmp.cmp(rpath, apath):
        return True

    apath.rename(failpath)

    showStringDiff(
        rpath.read_text(),
        failpath.read_text(),
        fromfile=rpath.name,
        tofile=failpath.name,
    )
    return False
