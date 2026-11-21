"""Build (or load) the CUDA extension.  Set LES_PTXAS_V=1 for register/spill counts.

    run.bat tools\build_ext.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels  # noqa

kernels.load_extension(verbose=True)
print("extension:", kernels.extension_status())
