"""Phase 16 efficiency tooling, kept outside the frozen v1 modules.

The v1 code (`v1.V1_PATHS`) was not edited in Phase 16, when it was checked against
``phase14-eval``; it is now checked against ``v1-code`` (`v1.V1_TAG`). Everything the efficiency
measurements need beyond it lives here:

- `v1`: refuse a run unless the v1 code paths still equal the tag's;
- `threads`: thread counts set before torch loads, and ONNX Runtime sessions with recorded options;
- `conditions`: machine and GPU state at the start and end of every profile;
- `onnx_build`: v1's pipeline objects running on sessions made by `threads.ort_session`.

Nothing is imported here, so ``glypharis.efficiency.threads`` can be imported before torch.
"""
