"""One lock for loading the large ML libraries (torch, transformers, sentence-transformers).

`transformers` imports its submodules lazily. If two threads trigger that at the same moment (for example the
start-up warm-up thread and a request that needs DistilBERT), one of them can see a half-initialised module and fail
with "cannot import name ...". Every place that imports or loads one of these libraries takes this lock first, so
only one thread at a time does the first import. It is re-entrant and always the innermost lock, so it cannot
deadlock with the per-model locks.
"""

import threading

HEAVY_IMPORT_LOCK = threading.RLock()
