#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Fri Feb 13 2026
Last version on Tue Aug 18 2026

@author: begum@mit.edu

Interactive fit tool for DARM_ERR strain ASD with gwinc quantum model + inferred classical residuals,
including a zoom panel. Parameter values are controlled with the sliders or typed in.
Recompute Model' button recomputes model curves and residuals.
Xlim and Ylim can be set for individual plots.

This version keeps the original working fitting-interface engine/layout,
but changes only the plot colors to the blaze palette:
    - no_sqz data/residual: dark navy
    - no_sqz model: grey
    - FIS: orange
    - FDS: blue

Important implementation detail:
    - Each slider has an internal key and a display label.
    - The display label can use LaTeX, e.g. $\\phi$.
    - The model still looks up parameters using the stable internal key.
"""

import os
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from matplotlib.backends.qt_compat import QtCore, QtWidgets

try:
    import yaml
except Exception:
    yaml = None

from utils import (
    load_calibrated_darm_err_mat,
    load_unwrapped_oltf,
    chi_mag_interp,
    finite_pos,
    pad_limits,
    safe_float,
)

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# Keep the original plot/slider physical width fixed, but make the figure
# wider to create a right-hand button/control panel.
# -------------------------------------------------------------------------
OLD_FIG_W = 13.2
NEW_FIG_W = 17.2
X_SCALE = OLD_FIG_W / NEW_FIG_W


def sx(x):
    return float(x) * X_SCALE


def sw(w):
    return float(w) * X_SCALE


class ParamRow:
    def __init__(self, fig, y, key, label, v0, vmin, vmax, step0, on_change):
        self.key = key
        self.label = label
        self.on_change = on_change
        self._sync_guard = False

        # These are scaled so the slider area keeps its original physical width
        # after the figure is widened.
        label_x = sx(0.185)
        slider_x = sx(0.20)
        slider_w = sw(0.51)

        val_x = sx(0.72)
        val_w = sw(0.075)

        minus_x = sx(0.805)
        plus_x = sx(0.838)
        step_x = sx(0.875)
        step_w = sw(0.07)
        h = 0.0175

        fig.text(label_x, y + h / 2, label, fontsize=9, ha="right", va="center")

        ax_s = fig.add_axes([slider_x, y, slider_w, h])
        self.slider = Slider(ax_s, "", vmin, vmax, valinit=v0, dragging=False)
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._on_slider)

        ax_v = fig.add_axes([val_x, y, val_w, h])
        self.valbox = TextBox(ax_v, "", initial=f"{v0:.6g}")
        self.valbox.text_disp.set_fontsize(8)
        self.valbox.on_submit(self._on_valbox_submit)

        ax_m = fig.add_axes([minus_x, y, sw(0.028), h])
        self.btn_minus = Button(ax_m, "−")
        self.btn_minus.on_clicked(self._step_minus)

        ax_p = fig.add_axes([plus_x, y, sw(0.028), h])
        self.btn_plus = Button(ax_p, "+")
        self.btn_plus.on_clicked(self._step_plus)

        ax_t = fig.add_axes([step_x, y, step_w, h])
        self.stepbox = TextBox(ax_t, "", initial=str(step0))
        self.stepbox.text_disp.set_fontsize(8)

    def _parse_step(self):
        v = safe_float(self.stepbox.text)
        if v is None or v <= 0:
            return None
        return v

    def _step_minus(self, _event):
        st = self._parse_step()
        if st is None:
            return
        self.slider.set_val(self.slider.val - st)

    def _step_plus(self, _event):
        st = self._parse_step()
        if st is None:
            return
        self.slider.set_val(self.slider.val + st)

    def _on_slider(self, val):
        if self._sync_guard:
            return

        self._sync_guard = True
        old_eventson = self.valbox.eventson
        self.valbox.eventson = False

        try:
            self.valbox.set_val(f"{float(val):.6g}")
        finally:
            self.valbox.eventson = old_eventson
            self._sync_guard = False

        if self.on_change is not None:
            self.on_change(self.key, float(val))

    def _on_valbox_submit(self, text):
        if self._sync_guard:
            return

        v = safe_float(text)

        if v is None:
            self._sync_guard = True
            try:
                self.valbox.set_val(f"{self.slider.val:.6g}")
            finally:
                self._sync_guard = False
            return

        v = float(np.clip(v, self.slider.valmin, self.slider.valmax))

        self._sync_guard = True

        try:
            self.slider.set_val(v)
            self.valbox.set_val(f"{v:.6g}")
        finally:
            self._sync_guard = False

        if self.on_change is not None:
            self.on_change(self.key, v)


class App:
    def __init__(
        self,
        yaml_path,
        data_mat,
        oltf_res_mat,
        oltf_nores_mat=None,
        keys=("no_sqz", "fis", "fds"),
        fmin=20,
        fmax=7000,
    ):
        self.yaml_path = str(yaml_path)
        self.data_mat = Path(data_mat)
        self.oltf_res_mat = Path(oltf_res_mat)
        self.oltf_nores_mat = Path(oltf_nores_mat) if oltf_nores_mat is not None else None
        self.keys = list(keys)

        self.fmin, self.fmax = fmin, fmax

        self.left_xmin0 = 20.0
        self.left_xmax0 = 4000.0

        self.zoom_xmin0 = 140.0
        self.zoom_xmax0 = 165.0

        self.ylim_tl = (5e-21, 2.5e-18)
        self.ylim_bl = (2e-21, 2.5e-19)
        self.ylim_tr = (4e-20, 2.2e-18)
        self.ylim_br = (5e-21, 3e-20)

        self.LO_ANG_DEG = -13.9
        self.arm_length = 3995.0
        self.PDeff = 0.925

        self.thermal_coat_A_m = 1.11e-20
        self.thermal_coat_alpha = 0.45
        self.thermal_coat_fref = 100.0

        self._did_unit_check = False
        self._dirty = False

        self.ymin0 = 1e-24 * self.arm_length
        self.ymax0 = 0.6e-21 * self.arm_length

        ff_meas, data_m = load_calibrated_darm_err_mat(self.data_mat, tuple(self.keys))
        self.ff_meas = ff_meas
        self.data = {k: np.asarray(data_m[k]) / self.arm_length for k in self.keys}

        self.ff_res, self.oltf_res = load_unwrapped_oltf(self.oltf_res_mat)

        if self.oltf_nores_mat is not None and self.oltf_nores_mat.exists():
            self.ff_nores, self.oltf_nores = load_unwrapped_oltf(self.oltf_nores_mat)
            self.using_nores_oltf = True
            print(f"Loaded no-res OLTF: {self.oltf_nores_mat}")
        else:
            self.ff_nores, self.oltf_nores = self.ff_res, self.oltf_res
            self.using_nores_oltf = False

            if self.oltf_nores_mat is not None:
                print(
                    "[warning] No-res OLTF file not found. "
                    "Using resonant OLTF also for the no-res denominator:"
                )
                print(f"          {self.oltf_nores_mat}")

        self.f_plot = np.geomspace(self.fmin, self.fmax, 1200)

        self.thermal_strain0 = (
            self.thermal_coat_A_m
            * (self.thermal_coat_fref / self.f_plot) ** self.thermal_coat_alpha
        ) / self.arm_length

        self.data_plot = {}

        for k in self.keys:
            y = self.data[k]
            m = (self.ff_meas > 0) & np.isfinite(y) & (y > 0)

            if np.sum(m) > 10:
                self.data_plot[k] = np.exp(
                    np.interp(
                        np.log(self.f_plot),
                        np.log(self.ff_meas[m]),
                        np.log(y[m]),
                    )
                )
            else:
                self.data_plot[k] = np.interp(self.f_plot, self.ff_meas, y)

        # Reduced figure height: was 9.4.
        self.fig = plt.figure(1, figsize=(NEW_FIG_W, 6))

        try:
            self.fig.canvas.manager.set_window_title("DARM_ERR SQZ Fit — displacement + inferred classical")
        except Exception:
            pass

        # Reduced window height: was frac_h=0.95.
        self._force_resize_window(frac_w=0.88, frac_h=1)

        TOP_GAP = 0.03
        BOTTOM_GAP = 0.0001
        INTERPLOT_GAP = 0.006

        AX_TOP_H = 0.30
        AX_BOT_H = 0.18

        X_LABEL_PAD = 0.4
        SLIDER_GAP_FROM_PLOT = 0.085

        GAP_SLIDERS_TO_CHECKBOX = 0.003
        GAP_CHECKBOX_TO_BUTTONS = 0.003

        BTN_H = 0.055
        SHOW_H = 0.030

        ax_top_bottom = 1.0 - TOP_GAP - AX_TOP_H
        ax_bot_bottom = ax_top_bottom - INTERPLOT_GAP - AX_BOT_H

        # Original plot geometry, scaled in x so the physical plot width stays fixed.
        old_left = 0.11
        old_right = 0.06
        old_total_w = 1.0 - old_left - old_right
        old_gap_mz = 0.02
        old_main_w = (old_total_w - old_gap_mz) * 3.0 / 4.0
        old_zoom_w = old_main_w / 3.0
        old_zoom_left = old_left + old_main_w + old_gap_mz

        left = sx(old_left)
        gap_mz = sw(old_gap_mz)
        main_w = sw(old_main_w)
        zoom_w = sw(old_zoom_w)
        zoom_left = sx(old_zoom_left)

        self.ax = self.fig.add_axes([left, ax_top_bottom, main_w, AX_TOP_H])
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlim(self.left_xmin0, self.left_xmax0)
        self.ax.set_ylabel("Observed displacement \n ASD [$\mathrm{m}/\sqrt{\mathrm{Hz}}$]")
        self.ax.grid(True, which="both", alpha=0.4)
        self.ax.set_ylim(*self.ylim_tl)
        self.ax.tick_params(axis="x", which="both", labelbottom=False)

        self.axz = self.fig.add_axes([zoom_left, ax_top_bottom, zoom_w, AX_TOP_H])
        self.axz.set_xscale("log")
        self.axz.set_yscale("log")
        self.axz.set_xlim(self.zoom_xmin0, self.zoom_xmax0)
        self.axz.set_ylim(*self.ylim_tr)
        self.axz.grid(True, which="major", alpha=0.40)
        self.axz.grid(True, which="minor", alpha=0.40)
        self.axz.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self.axz.yaxis.set_ticks_position("right")
        self.axz.yaxis.set_label_position("right")
        self.axz.tick_params(axis="y", which="both", right=True, labelright=True, left=False, labelleft=False)

        self.axc = self.fig.add_axes([left, ax_bot_bottom, main_w, AX_BOT_H])
        self.axc.set_xscale("log")
        self.axc.set_yscale("log")
        self.axc.set_xlim(self.left_xmin0, self.left_xmax0)
        self.axc.set_xlabel("Frequency [Hz]", labelpad=X_LABEL_PAD)
        self.axc.set_ylabel("Inferred classical \n ASD/ $|\\chi|$ [$\\mathrm{m}/\\sqrt{\\mathrm{Hz}}$]")
        self.axc.grid(True, which="both", alpha=0.4)
        self.axc.set_ylim(*self.ylim_bl)

        self.axcz = self.fig.add_axes([zoom_left, ax_bot_bottom, zoom_w, AX_BOT_H])
        self.axcz.set_xscale("log")
        self.axcz.set_yscale("log")
        self.axcz.set_xlim(self.zoom_xmin0, self.zoom_xmax0)
        self.axcz.set_ylim(*self.ylim_br)
        self.axcz.set_xlabel("Frequency [Hz]", labelpad=X_LABEL_PAD)
        self.axcz.grid(True, which="major", alpha=0.40)
        self.axcz.grid(True, which="minor", alpha=0.40)
        self.axcz.yaxis.set_ticks_position("right")
        self.axcz.yaxis.set_label_position("right")
        self.axcz.tick_params(axis="y", which="both", right=True, labelright=True, left=False, labelleft=False)

        self._update_zoom_xticks()

        self.colors = {
            "no_sqz": (0.0000, 0.0000, 0.1724),
            "fis": (0.9655, 0.5172, 0.0345),
            "fds": (0.4000, 0.6000, 0.9000),
        }

        self.model_colors = dict(self.colors)
        self.model_colors["no_sqz"] = "0.4"

        self.lines_data = {}
        self.lines_model = {}
        self.lines_classical = {}
        self.lines_data_z = {}
        self.lines_model_z = {}
        self.lines_classical_z = {}

        for k in self.keys:
            yd_m = self.data_plot[k] * self.arm_length

            (ld,) = self.ax.plot(
                self.f_plot,
                yd_m,
                color=self.colors[k],
                alpha=1.0,
                lw=1.15,
                label=f"data {k}",
            )
            self.lines_data[k] = ld

            (ldz,) = self.axz.plot(
                self.f_plot,
                yd_m,
                color=self.colors[k],
                alpha=1.0,
                lw=1.15,
            )
            self.lines_data_z[k] = ldz

            (lm,) = self.ax.plot(
                self.f_plot,
                np.nan * np.ones_like(self.f_plot),
                color=self.model_colors[k],
                alpha=1.0,
                lw=1.60,
                ls="--",
                label=f"model {k}",
            )
            self.lines_model[k] = lm

            (lmz,) = self.axz.plot(
                self.f_plot,
                np.nan * np.ones_like(self.f_plot),
                color=self.model_colors[k],
                alpha=1.0,
                lw=1.60,
                ls="--",
            )
            self.lines_model_z[k] = lmz

            (lc,) = self.axc.plot(
                self.f_plot,
                np.nan * np.ones_like(self.f_plot),
                color=self.colors[k],
                alpha=1.0,
                lw=1.15,
                ls="-",
                label=f"{k}",
            )
            self.lines_classical[k] = lc

            (lcz,) = self.axcz.plot(
                self.f_plot,
                np.nan * np.ones_like(self.f_plot),
                color=self.colors[k],
                alpha=1.0,
                lw=1.15,
                ls="-",
            )
            self.lines_classical_z[k] = lcz

        (self.line_thermal,) = self.axc.plot(
            self.f_plot,
            np.nan * np.ones_like(self.f_plot),
            color="0.25",
            lw=1.20,
            alpha=1.0,
            ls="--",
            label="thermal coat",
        )

        (self.line_thermal_z,) = self.axcz.plot(
            self.f_plot,
            np.nan * np.ones_like(self.f_plot),
            color="0.25",
            lw=1.20,
            alpha=1.0,
            ls="--",
        )

        self.lim_target = "Left"
        self.target_btns = {}
        self.target_btn_axes = {}

        self.chk_axes = []
        self.chk_buttons = []

        self.rows = {}

        self.btn_recompute = None
        self.btn_save_yaml = None
        self.btn_import_yaml = None
        self.btn_rescale_y = None
        self.btn_rescale_x = None
        self.btn_apply_limits = None

        self.box_xmin = None
        self.box_xmax = None
        self.box_ymin = None
        self.box_ymax = None

        self._controls_anchor = dict(
            ax_bot_bottom=ax_bot_bottom,
            bottom_gap=BOTTOM_GAP,
            slider_gap_from_plot=SLIDER_GAP_FROM_PLOT,
            gap_sliders_to_checkbox=GAP_SLIDERS_TO_CHECKBOX,
            gap_checkbox_to_buttons=GAP_CHECKBOX_TO_BUTTONS,
            btn_h=BTN_H,
            show_h=SHOW_H,
        )

        self._build_controls()
        self._sync_limit_boxes_from_axes()

        self.recompute()
        self._refresh_legends()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _refresh_legends(self):
        legend_items_top = [
            (self.lines_data["no_sqz"], "No SQZ data"),
            (self.lines_data["fis"], "FIS data"),
            (self.lines_data["fds"], "FDS data"),
            (self.lines_model["no_sqz"], "No SQZ model"),
            (self.lines_model["fis"], "FIS model"),
            (self.lines_model["fds"], "FDS model"),
        ]

        hh, ll = [], []

        for line, label in legend_items_top:
            if line.get_visible():
                hh.append(line)
                ll.append(label)

        self.ax.legend(
            hh,
            ll,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            fontsize=9,
            framealpha=0.85,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.45,
            borderaxespad=0.2,
        )

        legend_items_bottom = [
            (self.lines_classical["no_sqz"], "No SQZ"),
            (self.lines_classical["fis"], "FIS"),
            (self.lines_classical["fds"], "FDS"),
            (self.line_thermal, "CTN"),
        ]

        hh, ll = [], []

        for line, label in legend_items_bottom:
            if line.get_visible():
                hh.append(line)
                ll.append(label)

        self.axc.legend(
            hh,
            ll,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            fontsize=9,
            framealpha=0.85,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.45,
            borderaxespad=0.2,
        )

    def _axes_for_target(self):
        t = self.lim_target

        if t == "TL":
            return [self.ax]
        if t == "BL":
            return [self.axc]
        if t == "TR":
            return [self.axz]
        if t == "BR":
            return [self.axcz]
        if t == "Left":
            return [self.ax, self.axc]
        if t == "Right":
            return [self.axz, self.axcz]

        return [self.ax, self.axc]

    def _sync_limit_boxes_from_axes(self):
        ax_ref = self._axes_for_target()[0]
        xmin, xmax = ax_ref.get_xlim()
        ymin, ymax = ax_ref.get_ylim()

        if self.box_xmin is not None:
            self.box_xmin.set_val(f"{xmin:.6g}")
        if self.box_xmax is not None:
            self.box_xmax.set_val(f"{xmax:.6g}")
        if self.box_ymin is not None:
            self.box_ymin.set_val(f"{ymin:.3e}")
        if self.box_ymax is not None:
            self.box_ymax.set_val(f"{ymax:.3e}")

    def _set_lim_target(self, name):
        self.lim_target = str(name)

        for k, btn in self.target_btns.items():
            if k == self.lim_target:
                btn.color = "0.70"
                btn.hovercolor = "0.65"
                btn.ax.set_facecolor("0.70")
            else:
                btn.color = "0.94"
                btn.hovercolor = "0.88"
                btn.ax.set_facecolor("0.94")

        self._sync_limit_boxes_from_axes()
        self.fig.canvas.draw_idle()

    def _update_zoom_xticks(self):
        for ax in (self.axz, self.axcz):
            xmin, xmax = ax.get_xlim()

            if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
                continue

            x_major = np.arange(np.ceil(xmin / 10.0) * 10.0, xmax + 0.1, 10.0)
            x_minor = np.arange(np.ceil(xmin), xmax + 0.1, 2.0)

            ax.xaxis.set_major_locator(mticker.FixedLocator(x_major))
            ax.xaxis.set_minor_locator(mticker.FixedLocator(x_minor))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.0f} Hz"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

            ax.grid(True, which="major", alpha=0.40)
            ax.grid(True, which="minor", alpha=0.40)

            ax.tick_params(axis="x", labelsize=9)

        self.axz.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    def _build_trace_checkboxes_panel(self, x, y_top, w, h=0.018, dy=0.024):
        items = [
            ("Nosqz", self.lines_data.get("no_sqz"), self.lines_data_z.get("no_sqz")),
            ("FIS", self.lines_data.get("fis"), self.lines_data_z.get("fis")),
            ("FDS", self.lines_data.get("fds"), self.lines_data_z.get("fds")),
            ("Mdl Nosqz", self.lines_model.get("no_sqz"), self.lines_model_z.get("no_sqz")),
            ("Mdl FIS", self.lines_model.get("fis"), self.lines_model_z.get("fis")),
            ("Mdl FDS", self.lines_model.get("fds"), self.lines_model_z.get("fds")),
            ("Res. Nosqz", self.lines_classical.get("no_sqz"), self.lines_classical_z.get("no_sqz")),
            ("Res FIS", self.lines_classical.get("fis"), self.lines_classical_z.get("fis")),
            ("Res. FDS", self.lines_classical.get("fds"), self.lines_classical_z.get("fds")),
            ("CTN", self.line_thermal, self.line_thermal_z),
        ]

        for ax in getattr(self, "chk_axes", []):
            try:
                ax.remove()
            except Exception:
                pass

        self.chk_axes = []
        self.chk_buttons = []

        ncols = 2
        col_gap = 0.012
        col_w = (w - col_gap) / ncols

        for i, (lab, line_left, line_right) in enumerate(items):
            col = i % ncols
            row = i // ncols

            xx = x + col * (col_w + col_gap)
            yy = y_top - row * dy

            ax = self.fig.add_axes([xx, yy, col_w, h])
            ax.set_facecolor("none")

            visible = bool(line_left.get_visible()) if line_left is not None else False
            cb = CheckButtons(ax, [lab], [visible])
            cb.labels[0].set_fontsize(8)

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            def _on(_label, ln=line_left, lnr=line_right):
                if ln is None:
                    return

                new_vis = not ln.get_visible()
                ln.set_visible(new_vis)

                if lnr is not None:
                    lnr.set_visible(new_vis)

                self._refresh_legends()
                self.fig.canvas.draw_idle()

            cb.on_clicked(_on)

            self.chk_axes.append(ax)
            self.chk_buttons.append(cb)

    def _force_resize_window(self, frac_w=0.52, frac_h=0.82):
        try:
            manager = plt.get_current_fig_manager()
            win = manager.window
            win.setWindowState(QtCore.Qt.WindowNoState)
            win.showNormal()
            screen = win.screen()
            geom = screen.availableGeometry()

            W = int(frac_w * geom.width())
            H = int(frac_h * geom.height())
            X = geom.x() + int((1 - frac_w) / 2 * geom.width())
            Y = geom.y() + int((1 - frac_h) / 2 * geom.height())

            win.resize(W, H)
            win.move(X, Y)

        except Exception:
            pass

    def _build_right_control_panel(self):
        panel_x = sx(1.00) + 0.018
        panel_w = 1.0 - panel_x - 0.018

        panel_ax = self.fig.add_axes([panel_x, 0.055, panel_w, 0.90])
        panel_ax.set_facecolor("0.965")
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])

        for spine in panel_ax.spines.values():
            spine.set_edgecolor("0.75")

        x = panel_x + 0.014
        w = panel_w - 0.028
        btn_h = 0.032

        self.fig.text(
            x,
            0.925,
            "Actions",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax_recompute = self.fig.add_axes([x, 0.880, w, btn_h])
        self.btn_recompute = Button(ax_recompute, "Recompute Model")
        self.btn_recompute.label.set_fontsize(8)
        self.btn_recompute.on_clicked(lambda _evt: self.recompute())

        ax_save = self.fig.add_axes([x, 0.842, w, btn_h])
        self.btn_save_yaml = Button(ax_save, "Save YAML")
        self.btn_save_yaml.label.set_fontsize(8)
        self.btn_save_yaml.on_clicked(lambda _evt: self.export_gwinc_params())

        ax_import = self.fig.add_axes([x, 0.804, w, btn_h])
        self.btn_import_yaml = Button(ax_import, "Import YAML")
        self.btn_import_yaml.label.set_fontsize(8)
        self.btn_import_yaml.on_clicked(lambda _evt: self.load_gwinc_yaml())

        ax_ry = self.fig.add_axes([x, 0.766, w, btn_h])
        self.btn_rescale_y = Button(ax_ry, "Rescale Y")
        self.btn_rescale_y.label.set_fontsize(8)
        self.btn_rescale_y.on_clicked(lambda _evt: self.rescale_y())

        ax_rx = self.fig.add_axes([x, 0.728, w, btn_h])
        self.btn_rescale_x = Button(ax_rx, "Rescale X")
        self.btn_rescale_x.label.set_fontsize(8)
        self.btn_rescale_x.on_clicked(lambda _evt: self.rescale_x())

        self.fig.text(
            x,
            0.670,
            "Plot limits",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

        box_h = 0.028
        label_w = 0.040
        box_w = w - label_w - 0.010
        row_gap = 0.035
        box_x = x + label_w + 0.010

        rows = [
            ("Xmin", "box_xmin", self.left_xmin0),
            ("Xmax", "box_xmax", self.left_xmax0),
            ("Ymin", "box_ymin", f"{self.ylim_tl[0]:.3e}"),
            ("Ymax", "box_ymax", f"{self.ylim_tl[1]:.3e}"),
        ]

        y0 = 0.625

        for i, (lab, attr, init) in enumerate(rows):
            yy = y0 - i * row_gap

            self.fig.text(
                x,
                yy + box_h / 2,
                lab,
                fontsize=8,
                ha="left",
                va="center",
            )

            ax_box = self.fig.add_axes([box_x, yy, box_w, box_h])
            tb = TextBox(ax_box, "", initial=str(init))
            tb.text_disp.set_fontsize(8)

            setattr(self, attr, tb)

        ax_apply = self.fig.add_axes([x, 0.472, w, btn_h])
        self.btn_apply_limits = Button(ax_apply, "Apply")
        self.btn_apply_limits.label.set_fontsize(8)
        self.btn_apply_limits.on_clicked(lambda _evt: self.apply_limits())

        self.fig.text(
            x,
            0.427,
            "Apply limits to:",
            fontsize=8,
            ha="left",
            va="center",
        )

        labels = ["TL", "BL", "TR", "BR", "Left", "Right"]
        targ_h = 0.026
        targ_w = (w - 0.010) / 2.0
        targ_gap_x = 0.010
        targ_gap_y = 0.009

        x_left = x
        x_right = x + targ_w + targ_gap_x
        y_start = 0.390

        for i, lab in enumerate(labels):
            col = i % 2
            row = i // 2

            xx = x_left if col == 0 else x_right
            yy = y_start - row * (targ_h + targ_gap_y)

            axb = self.fig.add_axes([xx, yy, targ_w, targ_h])
            axb.set_facecolor("0.94")

            btn = Button(axb, lab, color="0.94", hovercolor="0.88")
            btn.label.set_fontsize(8)

            def _mk_cb(name):
                return lambda _evt: self._set_lim_target(name)

            btn.on_clicked(_mk_cb(lab))

            self.target_btns[lab] = btn
            self.target_btn_axes[lab] = axb

        self.fig.text(
            x,
            0.275,
            "Show traces",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

        self._build_trace_checkboxes_panel(
            x=x,
            y_top=0.248,
            w=w,
            h=0.018,
            dy=0.024,
        )

        self._set_lim_target(self.lim_target)

    def _build_controls(self):
        params = [
            ("Arm Power", "Arm Power", 269572.0, 0, 320e3, 1e3),
            ("SEC Detuning (deg)", "SEC Detuning (deg)", -0.0414, -1, 1, 0.001),
            ("Injected Squeezing (dB)", "Injected Squeezing (dB)", 18.1414, 0, 20, 0.05),
            ("Injection Loss", "Injection Loss", 0.05, 0, 1, 0.002),
            ("FC Detuning (Hz)", "FC Detuning (Hz)", -27.26, -40, 50, 0.05),
            ("FC Mismatch", "FC Mismatch $\\%$", 0.024, 0, 0.1, 0.0005),
            ("FC Mismatch Phase (rad)", "FC Mismatch $\\phi$ (rad)", 3.06, 0, 2 * np.pi, 0.01),
            ("IFO-OMC Mismatch", "IFO-OMC Mismatch $\\%$", 0.0114, 0, 0.2, 0.001),
            ("IFO-OMC Mismatch Phase (rad)", "IFO-OMC Mismatch $\\phi$ (rad)", 3.615, 0, 2 * np.pi, 0.01),
            ("SQZ-OMC Mismatch", "SQZ-OMC Mismatch $\\%$", 0.0212, 0, 0.2, 0.001),
            ("SQZ-OMC Mismatch Phase (rad)", "SQZ-OMC Mismatch $\\phi$ (rad)", 0.3145, 0, 2 * np.pi, 0.01),
            ("Phase Noise (rad)", "Phase Noise (rad)", 0.01984, 0, 0.5, 0.001),
            ("OLTF Gain (with res)", "OLTF Gain (with res)", 1.0018, 0.98, 1.05, 0.0005),
            ("OLTF Gain (no res)", "OLTF Gain (no res)", 1.002, 0.98, 1.05, 0.0005),
        ]

        ax_bot_bottom = self._controls_anchor["ax_bot_bottom"]
        slider_gap_from_plot = self._controls_anchor["slider_gap_from_plot"]

        y_top = ax_bot_bottom - slider_gap_from_plot
        dy = 0.023

        def on_change(_key, _val):
            self._dirty = True

        for i, (key, label, v0, vmin, vmax, step0) in enumerate(params):
            y = y_top - i * dy
            self.rows[key] = ParamRow(self.fig, y, key, label, v0, vmin, vmax, step0, on_change)

        self.fig.text(sx(0.885), y_top + 0.018, "step", fontsize=8, ha="left", va="bottom")
        self.fig.text(sx(0.72), y_top + 0.018, "value", fontsize=8, ha="left", va="bottom")

        self._build_right_control_panel()

    def _set_yaml_value(self, cfg, path, value):
        d = cfg

        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, np.ndarray):
            value = value.tolist()

        d[path[-1]] = value

    def _get_yaml_value(self, cfg, path, default=None):
        d = cfg

        for key in path:
            if not isinstance(d, dict):
                return default
            if key not in d:
                return default
            d = d[key]

        return d

    def _slider_values_from_yaml(self, cfg):
        vals = {}

        v = self._get_yaml_value(cfg, ("Laser", "ArmPower"))
        if v is not None:
            vals["Arm Power"] = float(v)

        v = self._get_yaml_value(cfg, ("Optics", "SRM", "Tunephase"))
        if v is not None:
            vals["SEC Detuning (deg)"] = float(v) * 180.0 / np.pi

        v = self._get_yaml_value(cfg, ("Squeezer", "AmplitudedB"))
        if v is not None:
            vals["Injected Squeezing (dB)"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "InjectionLoss"))
        if v is not None:
            vals["Injection Loss"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "FilterCavity", "fdetune"))
        if v is not None:
            vals["FC Detuning (Hz)"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "FilterCavity", "L_mm"))
        if v is not None:
            vals["FC Mismatch"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "FilterCavity", "psi_mm"))
        if v is not None:
            vals["FC Mismatch Phase (rad)"] = float(v)

        v = self._get_yaml_value(cfg, ("Optics", "MM_IFO_OMC"))
        if v is not None:
            vals["IFO-OMC Mismatch"] = float(v)

        v = self._get_yaml_value(cfg, ("Optics", "MM_IFO_OMCphi"))
        if v is not None:
            vals["IFO-OMC Mismatch Phase (rad)"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "MM_SQZ_OMC"))
        if v is not None:
            vals["SQZ-OMC Mismatch"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "MM_SQZ_OMCphi"))
        if v is not None:
            vals["SQZ-OMC Mismatch Phase (rad)"] = float(v)

        v = self._get_yaml_value(cfg, ("Squeezer", "SQZAngleRMS"))
        if v is not None:
            vals["Phase Noise (rad)"] = float(v)

        v = self._get_yaml_value(cfg, ("FitUI", "OLTFGainWithRes"))
        if v is not None:
            vals["OLTF Gain (with res)"] = float(v)

        v = self._get_yaml_value(cfg, ("FitUI", "OLTFGainNoRes"))
        if v is not None:
            vals["OLTF Gain (no res)"] = float(v)

        return vals

    def _apply_current_params_to_yaml(self, cfg):
        self._set_yaml_value(cfg, ("Laser", "ArmPower"), float(self.v("Arm Power")))

        self._set_yaml_value(
            cfg,
            ("Optics", "SRM", "Tunephase"),
            float(np.pi * self.v("SEC Detuning (deg)") / 180.0),
        )

        self._set_yaml_value(
            cfg,
            ("Optics", "Quadrature", "dc"),
            float(np.pi / 2 + self.LO_ANG_DEG * np.pi / 180),
        )

        self._set_yaml_value(cfg, ("Optics", "PhotoDetectorEfficiency"), float(self.PDeff))
        self._set_yaml_value(cfg, ("Infrastructure", "Length"), float(self.arm_length))

        self._set_yaml_value(cfg, ("Squeezer", "Type"), "Freq Dependent")
        self._set_yaml_value(cfg, ("Squeezer", "AmplitudedB"), float(self.v("Injected Squeezing (dB)")))
        self._set_yaml_value(cfg, ("Squeezer", "InjectionLoss"), float(self.v("Injection Loss")))
        self._set_yaml_value(cfg, ("Squeezer", "SQZAngleRMS"), float(self.v("Phase Noise (rad)")))
        self._set_yaml_value(cfg, ("Squeezer", "SQZAngle"), float((0.0 - self.LO_ANG_DEG) * np.pi / 180))

        self._set_yaml_value(cfg, ("Squeezer", "FilterCavity", "fdetune"), float(self.v("FC Detuning (Hz)")))
        self._set_yaml_value(cfg, ("Squeezer", "FilterCavity", "L_mm"), float(self.v("FC Mismatch")))

        self._set_yaml_value(
            cfg,
            ("Squeezer", "FilterCavity", "psi_mm"),
            float(self.v("FC Mismatch Phase (rad)")),
        )

        self._set_yaml_value(cfg, ("Optics", "MM_IFO_OMC"), float(self.v("IFO-OMC Mismatch")))

        self._set_yaml_value(
            cfg,
            ("Optics", "MM_IFO_OMCphi"),
            float(self.v("IFO-OMC Mismatch Phase (rad)")),
        )

        self._set_yaml_value(cfg, ("Squeezer", "MM_SQZ_OMC"), float(self.v("SQZ-OMC Mismatch")))

        self._set_yaml_value(
            cfg,
            ("Squeezer", "MM_SQZ_OMCphi"),
            float(self.v("SQZ-OMC Mismatch Phase (rad)")),
        )

        self._set_yaml_value(cfg, ("FitUI", "OLTFGainWithRes"), float(self.v("OLTF Gain (with res)")))
        self._set_yaml_value(cfg, ("FitUI", "OLTFGainNoRes"), float(self.v("OLTF Gain (no res)")))

        return cfg

    def _set_sliders_from_values(self, vals):
        for key, val in vals.items():
            if key not in self.rows:
                continue

            row = self.rows[key]
            val = float(np.clip(float(val), row.slider.valmin, row.slider.valmax))
            row.slider.set_val(val)

    def export_gwinc_params(self):
        if yaml is None:
            print("[error] PyYAML is not available, so YAML export cannot run.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"gwinc_fds_fit_params_{timestamp}.yaml"

        outpath_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save updated GWINC YAML",
            str(Path(__file__).resolve().parent / default_name),
            "YAML files (*.yaml *.yml);;All files (*)",
        )

        if not outpath_str:
            print("GWINC YAML export canceled.")
            return

        outpath = Path(outpath_str)

        with open(self.yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            cfg = {}

        cfg = self._apply_current_params_to_yaml(cfg)

        with open(outpath, "w") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

        print(f"Exported updated FDS GWINC YAML to: {outpath}")

    def load_gwinc_yaml(self):
        if yaml is None:
            print("[error] PyYAML is not available, so YAML loading cannot run.")
            return

        inpath_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Import GWINC YAML",
            str(Path(self.yaml_path).resolve().parent),
            "YAML files (*.yaml *.yml);;All files (*)",
        )

        if not inpath_str:
            print("GWINC YAML import canceled.")
            return

        inpath = Path(inpath_str)

        with open(inpath, "r") as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            cfg = {}

        if not isinstance(cfg, dict):
            print(f"[error] YAML file did not load as a dictionary: {inpath}")
            return

        self.yaml_path = str(inpath)

        qdc = self._get_yaml_value(cfg, ("Optics", "Quadrature", "dc"))
        if qdc is not None:
            self.LO_ANG_DEG = float((float(qdc) - np.pi / 2.0) * 180.0 / np.pi)

        pdeff = self._get_yaml_value(cfg, ("Optics", "PhotoDetectorEfficiency"))
        if pdeff is not None:
            self.PDeff = float(pdeff)

        vals = self._slider_values_from_yaml(cfg)
        self._set_sliders_from_values(vals)

        print(f"Imported GWINC YAML from: {inpath}")
        print(f"Updated {len(vals)} slider values.")

        self.recompute()

    def v(self, name):
        return self.rows[name].slider.val

    def _make_budget(self, freq):
        import gwinc

        budget = gwinc.load_budget(self.yaml_path, freq, bname="Quantum")
        ifo = budget.ifo

        ifo.Optics.Quadrature.dc = np.pi / 2 + self.LO_ANG_DEG * np.pi / 180
        ifo.Optics.PhotoDetectorEfficiency = self.PDeff
        ifo.Infrastructure.Length = self.arm_length

        return budget

    def _run_model_quantum(self, key):
        budget = self._make_budget(self.f_plot)
        ifo = budget.ifo

        ifo.Laser.ArmPower = self.v("Arm Power")
        ifo.Optics.SRM.Tunephase = np.pi * self.v("SEC Detuning (deg)") / 180.0

        key = key.lower()

        if key == "no_sqz":
            if hasattr(ifo, "Squeezer"):
                try:
                    del ifo.Squeezer.FilterCavity
                except Exception:
                    pass

                try:
                    del ifo.Squeezer
                except Exception:
                    pass

        else:
            ifo.Squeezer.AmplitudedB = self.v("Injected Squeezing (dB)")
            ifo.Squeezer.InjectionLoss = self.v("Injection Loss")
            ifo.Squeezer.SQZAngleRMS = self.v("Phase Noise (rad)")

            if key == "fis":
                ifo.Squeezer.Type = "Freq Independent"
            elif key == "fds":
                ifo.Squeezer.Type = "Freq Dependent"

            ifo.Squeezer.FilterCavity.fdetune = self.v("FC Detuning (Hz)")
            ifo.Squeezer.FilterCavity.L_mm = self.v("FC Mismatch")
            ifo.Squeezer.FilterCavity.psi_mm = self.v("FC Mismatch Phase (rad)")

            ifo.Optics.MM_IFO_OMC = self.v("IFO-OMC Mismatch")
            ifo.Optics.MM_IFO_OMCphi = self.v("IFO-OMC Mismatch Phase (rad)")

            ifo.Squeezer.MM_SQZ_OMC = self.v("SQZ-OMC Mismatch")
            ifo.Squeezer.MM_SQZ_OMCphi = self.v("SQZ-OMC Mismatch Phase (rad)")

            ifo.Squeezer.SQZAngle = (0.0 - self.LO_ANG_DEG) * np.pi / 180

        tr = budget.run()
        asd = np.sqrt(tr.psd)

        if not self._did_unit_check:
            self._did_unit_check = True
            f0 = 100.0
            i0 = int(np.argmin(np.abs(self.f_plot - f0)))
            print(f"[unit check] gwinc sqrt(psd) @ {self.f_plot[i0]:.1f} Hz = {asd[i0]:.3e}")

        return asd

    def _model_as_strain(self, asd_from_gwinc):
        return asd_from_gwinc / self.arm_length

    def recompute(self):
        self._dirty = False

        for k in self.keys:
            self.lines_model[k].set_linestyle(":")
            self.lines_model[k].set_alpha(1.0)
            self.lines_model_z[k].set_linestyle(":")
            self.lines_model_z[k].set_alpha(1.0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        t0 = time.time()
        print("Running gwinc (model update)...")

        try:
            g_res = self.v("OLTF Gain (with res)")
            g_nores = self.v("OLTF Gain (no res)")

            chi_res = chi_mag_interp(self.f_plot, self.ff_res, self.oltf_res, g_res)
            chi_nores = chi_mag_interp(self.f_plot, self.ff_nores, self.oltf_nores, g_nores)

            chi_ratio = chi_res
            denom = np.abs(chi_nores)
            denom = np.where(np.isfinite(denom) & (denom > 0), denom, np.nan)

            base_asd = {k: self._run_model_quantum(k) for k in self.keys}

            for k in self.keys:
                yq_strain = self._model_as_strain(base_asd[k]) * chi_ratio
                yq_strain = np.where(np.isfinite(yq_strain) & (yq_strain > 0), yq_strain, np.nan)

                yq_m = yq_strain * self.arm_length

                self.lines_model[k].set_ydata(yq_m)
                self.lines_model[k].set_linestyle("--")
                self.lines_model[k].set_alpha(1.0)

                self.lines_model_z[k].set_ydata(yq_m)
                self.lines_model_z[k].set_linestyle("--")
                self.lines_model_z[k].set_alpha(1.0)

            for k in self.keys:
                yd_m = np.asarray(self.lines_data[k].get_ydata(), dtype=float)
                yq_m = np.asarray(self.lines_model[k].get_ydata(), dtype=float)

                ok = np.isfinite(yd_m) & (yd_m > 0) & np.isfinite(yq_m) & (yq_m > 0)
                c_m = np.full_like(yd_m, np.nan, dtype=float)

                if np.any(ok):
                    diff = yd_m[ok] ** 2 - yq_m[ok] ** 2
                    diff = np.maximum(diff, 0.0)
                    c_m[ok] = np.sqrt(diff)

                c_div_m = c_m / denom
                c_div_m = np.where(np.isfinite(c_div_m) & (c_div_m > 0), c_div_m, np.nan)

                self.lines_classical[k].set_ydata(c_div_m)
                self.lines_classical_z[k].set_ydata(c_div_m)

            y_th_div_m = np.where(
                np.isfinite(self.thermal_strain0) & (self.thermal_strain0 > 0),
                self.thermal_strain0 * self.arm_length,
                np.nan,
            )

            self.line_thermal.set_ydata(y_th_div_m)
            self.line_thermal_z.set_ydata(y_th_div_m)

            print(f"Done. ({time.time() - t0:.2f} s)")

            self._refresh_legends()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print("GWinc exception:", repr(e))

    def rescale_y(self, update_boxes=True):
        def _lims_from_lines(lines):
            vals = []

            for ln in lines:
                if ln is None:
                    continue

                if ln.get_visible():
                    vals.append(finite_pos(ln.get_ydata()))

            v = np.hstack([x for x in vals if x.size > 0]) if any(x.size > 0 for x in vals) else None

            if v is None or v.size == 0:
                return None

            lo = np.nanpercentile(v, 1)
            hi = np.nanpercentile(v, 99)

            return pad_limits(lo, hi, pad=0.20)

        t = self.lim_target

        def _top_lines(right=False):
            if not right:
                return [self.lines_data[k] for k in self.keys] + [self.lines_model[k] for k in self.keys]

            return [self.lines_data_z[k] for k in self.keys] + [self.lines_model_z[k] for k in self.keys]

        def _bot_lines(right=False):
            if not right:
                return [self.lines_classical[k] for k in self.keys] + [self.line_thermal]

            return [self.lines_classical_z[k] for k in self.keys] + [self.line_thermal_z]

        if t == "TL":
            lims = _lims_from_lines(_top_lines(False))
            if lims is not None:
                self.ax.set_ylim(*lims)

        elif t == "TR":
            lims = _lims_from_lines(_top_lines(True))
            if lims is not None:
                self.axz.set_ylim(*lims)

        elif t == "BL":
            lims = _lims_from_lines(_bot_lines(False))
            if lims is not None:
                self.axc.set_ylim(*lims)

        elif t == "BR":
            lims = _lims_from_lines(_bot_lines(True))
            if lims is not None:
                self.axcz.set_ylim(*lims)

        elif t == "Left":
            lims = _lims_from_lines(_top_lines(False))
            if lims is not None:
                self.ax.set_ylim(*lims)

            lims = _lims_from_lines(_bot_lines(False))
            if lims is not None:
                self.axc.set_ylim(*lims)

        elif t == "Right":
            lims = _lims_from_lines(_top_lines(True))
            if lims is not None:
                self.axz.set_ylim(*lims)

            lims = _lims_from_lines(_bot_lines(True))
            if lims is not None:
                self.axcz.set_ylim(*lims)

        if update_boxes:
            self._sync_limit_boxes_from_axes()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def rescale_x(self, update_boxes=True):
        tgt_axes = self._axes_for_target()

        if self.lim_target in ("TL", "BL", "Left"):
            for ax in tgt_axes:
                ax.set_xlim(self.left_xmin0, self.left_xmax0)

        elif self.lim_target in ("TR", "BR", "Right"):
            for ax in tgt_axes:
                ax.set_xlim(self.zoom_xmin0, self.zoom_xmax0)

        else:
            for ax in tgt_axes:
                ax.set_xlim(self.fmin, self.fmax)

        if update_boxes:
            self._sync_limit_boxes_from_axes()

        if self.lim_target in ("TR", "BR", "Right"):
            self._update_zoom_xticks()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def apply_limits(self):
        xmin = safe_float(self.box_xmin.text)
        xmax = safe_float(self.box_xmax.text)
        ymin = safe_float(self.box_ymin.text)
        ymax = safe_float(self.box_ymax.text)

        tgt_axes = self._axes_for_target()

        if xmin is not None and xmax is not None and xmin > 0 and xmax > xmin:
            for ax in tgt_axes:
                ax.set_xlim(xmin, xmax)

        if ymin is not None and ymax is not None and ymin > 0 and ymax > ymin:
            for ax in tgt_axes:
                ax.set_ylim(ymin, ymax)

        if any(ax in (self.axz, self.axcz) for ax in tgt_axes):
            self._update_zoom_xticks()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    os.chdir(HERE)

    yaml_path = HERE / "april9.yaml"
    data_mat = HERE / "calibrated_darm_err_with_resonance.mat"
    oltf_res_mat = HERE / "unwrapped_oltf_with_resonance.mat"
    oltf_nores_mat = HERE / "unwrapped_oltf_no_resonance.mat"

    app = App(
        yaml_path=str(yaml_path),
        data_mat=data_mat,
        oltf_res_mat=oltf_res_mat,
        oltf_nores_mat=oltf_nores_mat,
        keys=("no_sqz", "fis", "fds"),
        fmin=20,
        fmax=7000,
    )

    plt.show()