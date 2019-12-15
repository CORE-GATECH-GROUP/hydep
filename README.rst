hydep
=====

A hybrid depletion framework for reactor physics applications and
analysis. 

Caveats / Warnings
------------------

This is a **highly experimental and developmental** library / tool.
While a modest set of cases are covered by tests and examples, there
are likely cases that are missed and could cause bugs. Please report
these and be forgiving.

This framework does not seek to be a general geometric modeling tool
for nuclear analysis, nor even a really good one. Some limits are
self imposed or assumed to move from developing geometry to
implementing the physics. First, the framework is primarily focused on
modeling Cartesian assemblies with annular fuel. This is sufficient for
most light water reactor analysis, as fuel pins are basically
concentric cylinders of materials, and these assemblies are regular
Cartesian lattices.

Second, many of the input files generated for transport solutions
should not be considered human readable. Some aesthetic considerations
have been taken, but universe names and identifiers may be hard to
understand. 
