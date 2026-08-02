# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

datas = []
binaries = []
hiddenimports = ['translate_ass_fast', 'translation_engine', 'subtitle_formats']

# Collect only essential PySide6 modules (not all)
hiddenimports += collect_submodules('PySide6.QtCore')
hiddenimports += collect_submodules('PySide6.QtGui')
hiddenimports += collect_submodules('PySide6.QtWidgets')
hiddenimports += ['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets']

# Inclui a DLL do runtime sem fixar uma versao especifica do Python.
python_dll_name = f'python{sys.version_info.major}{sys.version_info.minor}.dll'
python_dll_dirs = (os.path.dirname(sys.executable), sys.base_prefix)
for python_dll_dir in python_dll_dirs:
    python_dll = os.path.join(python_dll_dir, python_dll_name)
    if os.path.exists(python_dll):
        binaries.append((python_dll, '.'))
        break


a = Analysis(
    ['app_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TradutorASS-PySide',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
