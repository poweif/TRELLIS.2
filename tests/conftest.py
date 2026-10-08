import os

# tests/phase/ holds GPU + model-download scripts from the MeshAnything work, some of which run
# inference at import time. Only collect them on request.
collect_ignore = [] if os.environ.get("RUN_PHASE_TESTS") == "1" else ["phase"]
