This directory contains older versions of ONNX backend tests.
When ONNX updates versions (which happens pretty often still)
the backend unit tests are updated (at least sometimes),
and the old tests can fail with the latest ONNX node definition.

Since frontendframwerork-to-ONNX are (mostly?) external projects
to ONNX, they don't track these version changes as promptly as
it would nice for use end-users to have, we end up in the
situation that old ONNX versions are still much used.

In order to check that onnx2c doesn't generate wrong
code out of these older versions, those old test are copied here.

The subdirectories here indicate the ONNX version, but this
is not a strict rule, more like guidlines to about when
the contents of that directory was saved.


The tests are verbatime from the onnx repo. Licenced
under Apache 2.0 (I think, unless there is a separate license
somewhere for the tests that I missed), which I think means
its ok to use like this. 
