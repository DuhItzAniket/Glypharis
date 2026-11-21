# Sample page for `glypharis-check`

- `page.png`: one white 1240 x 1754 grey-scale page (A4 at 150 dpi) with eight lines of the
  project's own text about OCR and how GLYPHARIS reads a page. It was rendered by
  `scripts/make_sample_page.py render` with Pillow in Open Sans Regular, version 3.003
  (Copyright 2020 The Open Sans Project Authors, SIL Open Font License 1.1, from Google Fonts).
  Only the image is shipped; the font file is not.
- `expected.json`: the lines as rendered, and the output of the `glypharis-v1` bundle (manifest
  SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`) on that page with
  the torch backend on the CPU and 4 threads: the line texts in reading order, the joined text and
  the model version. It holds no timings. Made by `scripts/make_sample_page.py expected`.

`glypharis-check` checks that the installed bundle is the one the registry pins and that it
reproduces `expected.json` exactly on this page. That shows the engine and weights are the pinned
ones; it says nothing about accuracy. The page is not drawn from any dataset, so it is neither
validation nor test data.
