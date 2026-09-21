"""Standalone serving shims, each run by a foreign interpreter as a subprocess.

A shim's directory must contain ONLY shims: when Python runs
``python path/to/prism_shim.py`` the script's directory lands on ``sys.path``,
so a sibling named e.g. ``mlx_lm.py`` would shadow the real ``mlx_lm`` package
(spark's own ``runtimes/mlx_lm.py`` adapter is exactly such a sibling — this
directory exists because of that collision).
"""
