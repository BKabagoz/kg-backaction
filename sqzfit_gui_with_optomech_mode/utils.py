#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 2026

@author: begum@mit.edu
"""

# utils.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio


def _matvec(x):
    return np.asarray(x).reshape(-1)


def load_calibrated_darm_err_mat(path_mat: Path, keys=("no_sqz", "fis", "fds")):
    M = sio.loadmat(str(path_mat))
    ff = _matvec(M["ff_err"])
    D = np.asarray(M["darm_err_calib"])
    if D.shape != (len(keys), len(ff)):
        raise ValueError(f"Expected darm_err_calib shape ({len(keys)},{len(ff)}), got {D.shape}")
    data = {keys[i]: _matvec(D[i, :]) for i in range(len(keys))}
    return ff, data


def load_unwrapped_oltf(path_mat: Path):
    M = sio.loadmat(str(path_mat))
    ff = _matvec(M["ff"])
    new_mag = _matvec(M["new_mag"])
    new_phase = _matvec(M["new_phase"])
    amp = 10 ** (new_mag / 20.0)
    oltf = amp * (np.cos(np.deg2rad(new_phase)) + 1j * np.sin(np.deg2rad(new_phase)))
    return ff, oltf


# def chi_mag_interp(freq_target, ff_oltf, oltf, gain):
#     chi = 1.0 / (1.0 - gain * oltf)
#     re = np.interp(freq_target, ff_oltf, np.real(chi))
#     im = np.interp(freq_target, ff_oltf, np.imag(chi))
#     return np.abs(re + 1j * im)


def chi_mag_interp(freq_target, ff_oltf, oltf, gain):
    chi = 1.0 / (1.0 - float(gain) * oltf)
    chi_mag = np.abs(chi)
    return np.interp(freq_target, ff_oltf, chi_mag)


def finite_pos(y):
    y = np.asarray(y)
    return y[np.isfinite(y) & (y > 0)]


def pad_limits(lo, hi, pad=0.20):
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo):
        return None
    fac = np.exp(pad)
    return lo / fac, hi * fac


def safe_float(s):
    try:
        v = float(str(s).strip())
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def interp_asd_loglog(f_src, y_src, f_tgt, fill=np.nan):
    f_src = np.asarray(f_src)
    y_src = np.asarray(y_src)
    f_tgt = np.asarray(f_tgt)

    m = (f_src > 0) & np.isfinite(f_src) & np.isfinite(y_src) & (y_src > 0)
    if np.sum(m) < 5:
        return np.full_like(f_tgt, float(fill), dtype=float)

    fs = f_src[m]
    ys = y_src[m]
    o = np.argsort(fs)
    fs = fs[o]
    ys = ys[o]

    y_out = np.full_like(f_tgt, float(fill), dtype=float)
    inside = (f_tgt >= fs[0]) & (f_tgt <= fs[-1]) & (f_tgt > 0)
    if np.any(inside):
        y_out[inside] = np.exp(np.interp(np.log(f_tgt[inside]), np.log(fs), np.log(ys)))
    return y_out