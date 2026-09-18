#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
First version: Fri Feb 13 2026
This version: Wed Aug 24 2026

@author: begum@mit.edu

LLM usage: OpenAI GPT-5.5 Thinking was used to assist with code refinement, debugging, and optimization 
of the graphical user interface used for fitting. All AI-assisted code was reviewed and validated 
by the authors. All scientific model choices and parameter interpretation were made by the authors. 
The LLM was used for code organization and interface iteration and was not used as an independent 
scientific validation of the analysis.

Interactive fit tool for DARM_ERR strain ASD with gwinc quantum model + inferred classical residuals,
including two corresponding zoom panels. Parameter values are controlled with native Qt sliders/text boxes.
Recompute Model button recomputes model curves and residuals.
Xlim and Ylim can be set for individual plots.

This version embeds the Matplotlib plot canvas and Qt controls in one window:
    - top row: plots on the left, show traces + actions + plot limits on the right
    - bottom row: two-column slider controls spanning the full window width
"""

import os
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.ticker as mticker

from matplotlib.figure import Figure

try:
    import yaml
except Exception:
    yaml = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui

from utils import (
    load_calibrated_darm_err_mat,
    load_unwrapped_oltf,
    chi_mag_interp,
    finite_pos,
    pad_limits,
    safe_float,
)

warnings.filterwarnings("ignore")


QT_HORIZONTAL = getattr(QtCore.Qt, "Horizontal", QtCore.Qt.Orientation.Horizontal)
QT_VERTICAL = getattr(QtCore.Qt, "Vertical", QtCore.Qt.Orientation.Vertical)
QT_ALIGN_RIGHT = getattr(QtCore.Qt, "AlignRight", QtCore.Qt.AlignmentFlag.AlignRight)
QT_ALIGN_LEFT = getattr(QtCore.Qt, "AlignLeft", QtCore.Qt.AlignmentFlag.AlignLeft)
QT_ALIGN_CENTER = getattr(QtCore.Qt, "AlignCenter", QtCore.Qt.AlignmentFlag.AlignCenter)
QT_ALIGN_VCENTER = getattr(QtCore.Qt, "AlignVCenter", QtCore.Qt.AlignmentFlag.AlignVCenter)


class DefaultMarkSlider(QtWidgets.QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._default_int = None

        # Hand-tune this number if the default marker is vertically off.
        # Positive moves the marker DOWN.
        # Negative moves the marker UP.
        self._default_marker_y_offset = -1

    def set_default_int(self, val):
        self._default_int = int(val)
        self.update()

    def paintEvent(self, event):
        if self._default_int is not None:
            vmin = self.minimum()
            vmax = self.maximum()

            if vmax > vmin:
                frac = (self._default_int - vmin) / float(vmax - vmin)
                frac = max(0.0, min(1.0, frac))

                opt = QtWidgets.QStyleOptionSlider()
                self.initStyleOption(opt)

                try:
                    cc_slider = QtWidgets.QStyle.CC_Slider
                    sc_groove = QtWidgets.QStyle.SC_SliderGroove
                    sc_handle = QtWidgets.QStyle.SC_SliderHandle
                except AttributeError:
                    cc_slider = QtWidgets.QStyle.ComplexControl.CC_Slider
                    sc_groove = QtWidgets.QStyle.SubControl.SC_SliderGroove
                    sc_handle = QtWidgets.QStyle.SubControl.SC_SliderHandle

                groove_rect = self.style().subControlRect(
                    cc_slider,
                    opt,
                    sc_groove,
                    self,
                )

                handle_rect = self.style().subControlRect(
                    cc_slider,
                    opt,
                    sc_handle,
                    self,
                )

                x0 = groove_rect.left() + handle_rect.width() / 2.0
                x1 = groove_rect.right() - handle_rect.width() / 2.0
                x = x0 + frac * (x1 - x0)

                y_mid = groove_rect.center().y() + self._default_marker_y_offset
                y0 = y_mid - 7
                y1 = y_mid + 7

                painter = QtGui.QPainter(self)

                try:
                    painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
                except AttributeError:
                    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

                # Keep the default/YAML marker visible in both themes.
                # Light mode: black. Dark mode: light gray.
                try:
                    window_role = QtGui.QPalette.Window
                except AttributeError:
                    window_role = QtGui.QPalette.ColorRole.Window

                window_color = self.palette().color(window_role)
                if window_color.lightness() < 128:
                    marker_color = QtGui.QColor("#b0b0b0")
                else:
                    marker_color = QtGui.QColor("black")

                pen = QtGui.QPen(marker_color)
                pen.setWidth(2)
                painter.setPen(pen)

                painter.drawLine(
                    int(round(x)),
                    int(round(y0)),
                    int(round(x)),
                    int(round(y1)),
                )
                painter.end()

        super().paintEvent(event)


class ParamRowQt(QtWidgets.QWidget):
    def __init__(self, key, label, v0, vmin, vmax, step0, on_change=None, parent=None):
        super().__init__(parent)

        self.key = key
        self.display_label = label
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.on_change = on_change
        self._sync_guard = False
        self._nsteps = 10000

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(1)

        self.label_widget = QtWidgets.QLabel(label)
        self.label_widget.setAlignment(QT_ALIGN_RIGHT | QT_ALIGN_VCENTER)
        self.label_widget.setFixedWidth(180)

        self.slider = DefaultMarkSlider(QT_HORIZONTAL)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self._nsteps)
        self.slider.setFixedWidth(240)
        self.slider.setFixedHeight(25)
        self.slider.setTracking(True)
        self.slider.set_default_int(self._int_from_val(v0))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
            }

            QSlider::handle:horizontal {
                width: 1px;
                height: 1px;
                margin: -2px 0px;
            }
        """)

        self.valbox = QtWidgets.QLineEdit(f"{float(v0):.6g}")
        self.valbox.setFixedWidth(65)

        self.btn_minus = QtWidgets.QPushButton("−")
        self.btn_minus.setFixedWidth(35)

        self.btn_plus = QtWidgets.QPushButton("+")
        self.btn_plus.setFixedWidth(35)

        self.stepbox = QtWidgets.QLineEdit(str(step0))
        self.stepbox.setFixedWidth(60)

        layout.addWidget(self.label_widget)
        layout.addWidget(self.slider)
        layout.addWidget(self.valbox)
        layout.addWidget(self.btn_minus)
        layout.addWidget(self.btn_plus)
        layout.addWidget(self.stepbox)

        self.set_val(v0, emit=False)

        self.slider.valueChanged.connect(self._on_slider_int)
        self.valbox.editingFinished.connect(self._on_valbox_finished)
        self.btn_minus.clicked.connect(self._step_minus)
        self.btn_plus.clicked.connect(self._step_plus)

    def _int_from_val(self, val):
        val = float(np.clip(val, self.vmin, self.vmax))
        if self.vmax == self.vmin:
            return 0
        return int(round((val - self.vmin) / (self.vmax - self.vmin) * self._nsteps))

    def _val_from_int(self, ival):
        ival = int(np.clip(ival, 0, self._nsteps))
        if self.vmax == self.vmin:
            return self.vmin
        return self.vmin + (self.vmax - self.vmin) * ival / self._nsteps

    def value(self):
        return self._val_from_int(self.slider.value())

    def commit_text_value(self):
        self._on_valbox_finished()

    def set_default_val(self, val, set_current=True, emit=False):
        val = float(np.clip(val, self.vmin, self.vmax))
        self.slider.set_default_int(self._int_from_val(val))

        if set_current:
            self.set_val(val, emit=emit)

    def set_val(self, val, emit=True):
        val = float(np.clip(val, self.vmin, self.vmax))
        ival = self._int_from_val(val)

        self._sync_guard = True
        try:
            old_slider_block = self.slider.blockSignals(True)
            old_valbox_block = self.valbox.blockSignals(True)

            self.slider.setValue(ival)
            self.valbox.setText(f"{val:.6g}")

            self.slider.blockSignals(old_slider_block)
            self.valbox.blockSignals(old_valbox_block)
        finally:
            self._sync_guard = False

        if emit and self.on_change is not None:
            self.on_change(self.key, val)

    def _parse_step(self):
        v = safe_float(self.stepbox.text())
        if v is None or v <= 0:
            return None
        return float(v)

    def _step_minus(self):
        st = self._parse_step()
        if st is None:
            return
        self.set_val(self.value() - st, emit=True)

    def _step_plus(self):
        st = self._parse_step()
        if st is None:
            return
        self.set_val(self.value() + st, emit=True)

    def _on_slider_int(self, ival):
        if self._sync_guard:
            return

        val = self._val_from_int(ival)

        old_block = self.valbox.blockSignals(True)
        self.valbox.setText(f"{val:.6g}")
        self.valbox.blockSignals(old_block)

        if self.on_change is not None:
            self.on_change(self.key, val)

    def _on_valbox_finished(self):
        if self._sync_guard:
            return

        v = safe_float(self.valbox.text())
        if v is None:
            self.valbox.setText(f"{self.value():.6g}")
            return

        self.set_val(v, emit=True)


class SliderPanelQt(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()

        self.app = app
        self.rows = {}
        self.default_values = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel("  Model parameters")
        self.title_label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
        self.title_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.title_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(240)

        rows_widget = QtWidgets.QWidget()
        rows_layout = QtWidgets.QGridLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setHorizontalSpacing(2)
        rows_layout.setVerticalSpacing(1)

        params = [
            ("Arm Power", "Arm Power", 269572.0, 0, 320e3, 1e3),
            ("SEC Detuning (deg)", "SEC Detuning (deg)", -0.0414, -1, 1, 0.001),
            ("Injected Squeezing (dB)", "Injected Squeezing (dB)", 18.1414, 0, 20, 0.05),
            ("Injection Loss", "Injection Loss", 0.05, 0, 1, 0.002),
            ("FC Detuning (Hz)", "FC Detuning (Hz)", -27.26, -40, 50, 0.05),

            ("FC Mismatch", "FC Mismatch %", 0.024, 0, 0.1, 0.0005),
            ("FC Mismatch Phase (rad)", "FC Mismatch φ (rad)", 3.06, 0, 2 * np.pi, 0.01),

            ("IFO-OMC Mismatch", "IFO-OMC Mismatch %", 0.0114, 0, 0.2, 0.001),
            ("IFO-OMC Mismatch Phase (rad)", "IFO-OMC Mismatch φ (rad)", 3.615, 0, 2 * np.pi, 0.01),

            ("SQZ-OMC Mismatch", "SQZ-OMC Mismatch %", 0.0212, 0, 0.2, 0.001),
            ("SQZ-OMC Mismatch Phase (rad)", "SQZ-OMC Mismatch φ (rad)", 0.3145, 0, 2 * np.pi, 0.01),

            ("Phase Noise (rad)", "Phase Noise (rad)", 0.01984, 0, 0.5, 0.001),
            ("OLTF Gain (with res)", "OLTF Gain (with res)", 1.0018, 0.98, 1.05, 0.0005),
            ("OLTF Gain (no res)", "OLTF Gain (no res)", 1.002, 0.98, 1.05, 0.0005),
        ]

        ncols = 2
        nrows = int(np.ceil(len(params) / ncols))

        for i, (key, label, v0, vmin, vmax, step0) in enumerate(params):
            col = i // nrows
            row_i = i % nrows

            row = ParamRowQt(key, label, v0, vmin, vmax, step0, on_change=self._on_param_change)
            self.rows[key] = row
            self.default_values[key] = float(v0)
            rows_layout.addWidget(row, row_i, col)

        rows_layout.setColumnStretch(0, 1)
        rows_layout.setColumnStretch(1, 1)

        scroll.setWidget(rows_widget)
        main_layout.addWidget(scroll)

        self.app.rows = self.rows

    def _on_param_change(self, _key, _val):
        self.app._dirty = True

    def commit_all_text_values(self):
        for row in self.rows.values():
            row.commit_text_value()

    def set_defaults_from_values(self, values, set_current=True):
        for key, val in values.items():
            if key not in self.rows:
                continue

            row = self.rows[key]
            val = float(np.clip(float(val), row.vmin, row.vmax))

            self.default_values[key] = val
            row.set_default_val(val, set_current=set_current, emit=False)

        self.app._dirty = True

    def reset_defaults(self):
        for key, row in self.rows.items():
            if key in self.default_values:
                row.set_val(self.default_values[key], emit=False)

        self.app._dirty = True


class RightPanelQt(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()

        self.app = app
        self.trace_checkboxes = {}
        self.target_buttons = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(10)

        show_group = QtWidgets.QGroupBox("Show traces")
        show_layout = QtWidgets.QGridLayout(show_group)
        show_layout.setContentsMargins(8, 8, 8, 8)
        show_layout.setHorizontalSpacing(8)
        show_layout.setVerticalSpacing(4)

        trace_items = [
            ("Nosqz", self.app.lines_data.get("no_sqz"), self.app.lines_data_z.get("no_sqz")),
            ("FIS", self.app.lines_data.get("fis"), self.app.lines_data_z.get("fis")),
            ("FDS", self.app.lines_data.get("fds"), self.app.lines_data_z.get("fds")),
            ("Mdl Nosqz", self.app.lines_model.get("no_sqz"), self.app.lines_model_z.get("no_sqz")),
            ("Mdl FIS", self.app.lines_model.get("fis"), self.app.lines_model_z.get("fis")),
            ("Mdl FDS", self.app.lines_model.get("fds"), self.app.lines_model_z.get("fds")),
            ("Residue Nosqz", self.app.lines_classical.get("no_sqz"), self.app.lines_classical_z.get("no_sqz")),
            ("Residue FIS", self.app.lines_classical.get("fis"), self.app.lines_classical_z.get("fis")),
            ("Residue FDS", self.app.lines_classical.get("fds"), self.app.lines_classical_z.get("fds")),
            ("CTN", self.app.line_thermal, self.app.line_thermal_z),
        ]

        n_show_cols = 2

        for i, (lab, line_left, line_right) in enumerate(trace_items):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(bool(line_left.get_visible()) if line_left is not None else False)

            def _mk_trace_cb(ln=line_left, lnr=line_right):
                def _on_toggled(checked):
                    if ln is None:
                        return
                    ln.set_visible(bool(checked))
                    if lnr is not None:
                        lnr.set_visible(bool(checked))
                    self.app._refresh_legends()
                    self.app.canvas.draw_idle()
                    self.app.canvas.flush_events()
                return _on_toggled

            cb.toggled.connect(_mk_trace_cb())
            self.trace_checkboxes[lab] = cb

            row = i // n_show_cols
            col = i % n_show_cols
            show_layout.addWidget(cb, row, col)

        show_layout.setColumnStretch(0, 1)
        show_layout.setColumnStretch(1, 1)

        main_layout.addWidget(show_group)

        action_group = QtWidgets.QGroupBox("Actions")
        action_layout = QtWidgets.QVBoxLayout(action_group)
        action_layout.setContentsMargins(8, 8, 8, 8)
        action_layout.setSpacing(7)

        self.btn_recompute = QtWidgets.QPushButton("Recompute Model")
        self.btn_reset_defaults = QtWidgets.QPushButton("Reset Defaults")
        self.btn_load_yaml = QtWidgets.QPushButton("Load YAML")
        self.btn_export_params = QtWidgets.QPushButton("Export YAML")
        self.btn_rescale_x = QtWidgets.QPushButton("Rescale X")
        self.btn_rescale_y = QtWidgets.QPushButton("Rescale Y")
        self.btn_dark_mode = QtWidgets.QPushButton("Light Mode" if self.app.dark_mode else "Dark Mode")
        self.btn_dark_mode.setCheckable(True)
        self.btn_dark_mode.setChecked(self.app.dark_mode)

        for btn in (
            self.btn_recompute,
            self.btn_reset_defaults,
            self.btn_dark_mode,
        ):
            btn.setFixedSize(150, 28)

        for btn in (
            self.btn_load_yaml,
            self.btn_export_params,
            self.btn_rescale_x,
            self.btn_rescale_y,
        ):
            btn.setFixedSize(110, 28)

        self.btn_recompute.clicked.connect(self.app.recompute)
        self.btn_reset_defaults.clicked.connect(self.app.reset_defaults)
        self.btn_load_yaml.clicked.connect(self.app.load_gwinc_yaml)
        self.btn_export_params.clicked.connect(self.app.export_gwinc_params)
        self.btn_rescale_y.clicked.connect(self.app.rescale_y)
        self.btn_rescale_x.clicked.connect(self.app.rescale_x)
        self.btn_dark_mode.toggled.connect(self.app.set_dark_mode)
        self.app.dark_mode_button = self.btn_dark_mode

        yaml_button_row = QtWidgets.QHBoxLayout()
        yaml_button_row.setContentsMargins(0, 0, 0, 0)
        yaml_button_row.setSpacing(3)
        yaml_button_row.addStretch(1)
        yaml_button_row.addWidget(self.btn_load_yaml)
        yaml_button_row.addWidget(self.btn_export_params)
        yaml_button_row.addStretch(1)

        rescale_button_row = QtWidgets.QHBoxLayout()
        rescale_button_row.setContentsMargins(0, 0, 0, 0)
        rescale_button_row.setSpacing(1)
        rescale_button_row.addStretch(1)
        rescale_button_row.addWidget(self.btn_rescale_x)
        rescale_button_row.addWidget(self.btn_rescale_y)
        rescale_button_row.addStretch(1)

        action_layout.addWidget(self.btn_recompute, alignment=QT_ALIGN_CENTER)
        action_layout.addWidget(self.btn_reset_defaults, alignment=QT_ALIGN_CENTER)
        action_layout.addLayout(yaml_button_row)
        action_layout.addLayout(rescale_button_row)
        action_layout.addWidget(self.btn_dark_mode, alignment=QT_ALIGN_CENTER)

        main_layout.addWidget(action_group)

        limits_group = QtWidgets.QGroupBox("Plot limits")
        limits_layout = QtWidgets.QVBoxLayout(limits_group)
        limits_layout.setContentsMargins(8, 8, 8, 8)
        limits_layout.setSpacing(7)

        self.box_xmin = QtWidgets.QLineEdit()
        self.box_xmax = QtWidgets.QLineEdit()
        self.box_ymin = QtWidgets.QLineEdit()
        self.box_ymax = QtWidgets.QLineEdit()

        try:
            click_focus = QtCore.Qt.ClickFocus
        except AttributeError:
            click_focus = QtCore.Qt.FocusPolicy.ClickFocus

        for box in (self.box_xmin, self.box_xmax, self.box_ymin, self.box_ymax):
            box.setFocusPolicy(click_focus)

        for lab, box in [
            ("Xmin", self.box_xmin),
            ("Xmax", self.box_xmax),
            ("Ymin", self.box_ymin),
            ("Ymax", self.box_ymax),
        ]:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(lab)
            label.setFixedWidth(40)
            box.setFixedWidth(115)
            row.addWidget(label)
            row.addWidget(box)
            row.addStretch(1)
            limits_layout.addLayout(row)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self.app.apply_limits)
        limits_layout.addWidget(self.btn_apply, alignment=QT_ALIGN_CENTER)

        target_label = QtWidgets.QLabel("Apply limits to:")
        limits_layout.addWidget(target_label)

        target_grid = QtWidgets.QGridLayout()
        target_grid.setHorizontalSpacing(5)
        target_grid.setVerticalSpacing(5)

        labels = ["TL", "TR", "BL",  "BR", "Left", "Right"]

        for i, lab in enumerate(labels):
            btn = QtWidgets.QPushButton(lab)
            btn.setCheckable(True)
            btn.setFixedWidth(62)

            def _mk_cb(name):
                return lambda _checked: self.app._set_lim_target(name)

            btn.clicked.connect(_mk_cb(lab))
            self.target_buttons[lab] = btn
            target_grid.addWidget(btn, i // 2, i % 2)

        limits_layout.addLayout(target_grid)
        main_layout.addWidget(limits_group)

        main_layout.addStretch(1)

        self.app.box_xmin = self.box_xmin
        self.app.box_xmax = self.box_xmax
        self.app.box_ymin = self.box_ymin
        self.app.box_ymax = self.box_ymax
        self.app.target_btns = self.target_buttons

        self.app._set_lim_target(self.app.lim_target)
        self.app._sync_limit_boxes_from_axes()

    def set_dirty_visual(self, dirty):
        self.btn_recompute.setText("Recompute Model")

        if dirty:
            # Dirty state: deliberately highlight the button.
            self.btn_recompute.setStyleSheet("""
                QPushButton {
                    background-color: #2d8cff;
                    color: white;
                    border: 1px solid #2d8cff;
                    border-radius: 4px;
                    padding: 0px;
                }
            """)
        else:
            # Clean state: remove all custom styling so this button looks
            # exactly like the other native Qt buttons in the Actions panel.
            # Re-apply the app palette first so the normal text color is
            # restored correctly after the blue/white dirty state.
            self.btn_recompute.setStyleSheet("")
            self.btn_recompute.setPalette(self.app.palette())


class App(QtWidgets.QMainWindow):
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
        super().__init__()

        self.setWindowTitle("DARM_ERR SQZ Fit — displacement + inferred classical")

        self.yaml_path = str(yaml_path)
        self.data_mat = Path(data_mat)
        self.oltf_res_mat = Path(oltf_res_mat)
        self.oltf_nores_mat = Path(oltf_nores_mat) if oltf_nores_mat is not None else None
        self.keys = list(keys)

        self.fmin, self.fmax = fmin, fmax

        self.left_xmin0 = 20.0
        self.left_xmax0 = 4000.0

        self.zoom_xmin0 = 138.0
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

        self.__dirty = False
        self._did_unit_check = False
        self.dark_mode = True
        self.dark_mode_button = None

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

        # Keep all Matplotlib figure text at the same size as the
        # zoom-panel x tick labels and the legends.
        self.figure_fontsize = 10

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

        self.fig = Figure(figsize=(13.2, 8.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        self._build_plot_axes()

        self.lim_target = None

        self.rows = {}
        self.target_btns = {}
        self.box_xmin = None
        self.box_xmax = None
        self.box_ymin = None
        self.box_ymax = None

        self.slider_panel = SliderPanelQt(self)
        self.right_panel = RightPanelQt(self)

        top_splitter = QtWidgets.QSplitter(QT_HORIZONTAL)
        top_splitter.addWidget(self.canvas)
        top_splitter.addWidget(self.right_panel)
        top_splitter.setStretchFactor(0, 5)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([1200, 280])

        main_splitter = QtWidgets.QSplitter(QT_VERTICAL)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.slider_panel)
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 1)
        # main_splitter.setSizes([700, 260])
        main_splitter.setSizes([650, 260])

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(main_splitter)

        self.setCentralWidget(central)
        self.resize(1300, 850)

        self.recompute()
        self._refresh_legends()
        self.apply_plot_theme()

        self.canvas.draw_idle()
        self.canvas.flush_events()

    @property
    def _dirty(self):
        return self.__dirty

    @_dirty.setter
    def _dirty(self, val):
        self.__dirty = bool(val)
        panel = getattr(self, "right_panel", None)
        if panel is not None:
            panel.set_dirty_visual(self.__dirty)

    def _line_colors_for_theme(self):
        if self.dark_mode:
            colors = {
                "no_sqz": "white",
                "fis": (0.9655, 0.5172, 0.0345),
                "fds": (0.4000, 0.6000, 0.9000),
            }
            model_colors = dict(colors)
            model_colors["no_sqz"] = "0.82"
            thermal_color = "0.72"
        else:
            colors = {
                "no_sqz": (0.0000, 0.0000, 0.1724),
                "fis": (0.9655, 0.5172, 0.0345),
                "fds": (0.4000, 0.6000, 0.9000),
            }
            model_colors = dict(colors)
            model_colors["no_sqz"] = "0.4"
            thermal_color = "0.25"

        return colors, model_colors, thermal_color

    def set_dark_mode(self, checked):
        self.dark_mode = bool(checked)

        if self.dark_mode_button is not None:
            self.dark_mode_button.setText("Light Mode" if self.dark_mode else "Dark Mode")

        self.apply_plot_theme()

    def apply_ui_theme(self):
        """Change only Qt UI colors; do not alter widget geometry or styling."""
        palette = QtGui.QPalette()

        try:
            Window = QtGui.QPalette.Window
            WindowText = QtGui.QPalette.WindowText
            Base = QtGui.QPalette.Base
            AlternateBase = QtGui.QPalette.AlternateBase
            ToolTipBase = QtGui.QPalette.ToolTipBase
            ToolTipText = QtGui.QPalette.ToolTipText
            Text = QtGui.QPalette.Text
            Button = QtGui.QPalette.Button
            ButtonText = QtGui.QPalette.ButtonText
            BrightText = QtGui.QPalette.BrightText
            Highlight = QtGui.QPalette.Highlight
            HighlightedText = QtGui.QPalette.HighlightedText
        except AttributeError:
            Window = QtGui.QPalette.ColorRole.Window
            WindowText = QtGui.QPalette.ColorRole.WindowText
            Base = QtGui.QPalette.ColorRole.Base
            AlternateBase = QtGui.QPalette.ColorRole.AlternateBase
            ToolTipBase = QtGui.QPalette.ColorRole.ToolTipBase
            ToolTipText = QtGui.QPalette.ColorRole.ToolTipText
            Text = QtGui.QPalette.ColorRole.Text
            Button = QtGui.QPalette.ColorRole.Button
            ButtonText = QtGui.QPalette.ColorRole.ButtonText
            BrightText = QtGui.QPalette.ColorRole.BrightText
            Highlight = QtGui.QPalette.ColorRole.Highlight
            HighlightedText = QtGui.QPalette.ColorRole.HighlightedText

        if self.dark_mode:
            palette.setColor(Window, QtGui.QColor("#1e1e1e"))
            palette.setColor(WindowText, QtGui.QColor("#f2f2f2"))
            palette.setColor(Base, QtGui.QColor("#2b2b2b"))
            palette.setColor(AlternateBase, QtGui.QColor("#252525"))
            palette.setColor(ToolTipBase, QtGui.QColor("#2b2b2b"))
            palette.setColor(ToolTipText, QtGui.QColor("#f2f2f2"))
            palette.setColor(Text, QtGui.QColor("#f2f2f2"))
            palette.setColor(Button, QtGui.QColor("#2b2b2b"))
            palette.setColor(ButtonText, QtGui.QColor("#f2f2f2"))
            palette.setColor(BrightText, QtGui.QColor("white"))
            palette.setColor(Highlight, QtGui.QColor("#2d8cff"))
            palette.setColor(HighlightedText, QtGui.QColor("white"))
        else:
            palette.setColor(Window, QtGui.QColor("#f0f0f0"))
            palette.setColor(WindowText, QtGui.QColor("#111111"))
            palette.setColor(Base, QtGui.QColor("white"))
            palette.setColor(AlternateBase, QtGui.QColor("#f7f7f7"))
            palette.setColor(ToolTipBase, QtGui.QColor("white"))
            palette.setColor(ToolTipText, QtGui.QColor("#111111"))
            palette.setColor(Text, QtGui.QColor("#111111"))
            palette.setColor(Button, QtGui.QColor("#f0f0f0"))
            palette.setColor(ButtonText, QtGui.QColor("#111111"))
            palette.setColor(BrightText, QtGui.QColor("black"))
            palette.setColor(Highlight, QtGui.QColor("#2d8cff"))
            palette.setColor(HighlightedText, QtGui.QColor("white"))

        # Applying a palette changes colors only. It does not change widget
        # dimensions, layouts, margins, button geometry, or slider geometry.
        self.setPalette(palette)

        central = self.centralWidget()
        if central is not None:
            central.setPalette(palette)

        if hasattr(self, "right_panel"):
            self.right_panel.setPalette(palette)
            # Re-apply the recompute-button state using the new theme colors.
            self.right_panel.set_dirty_visual(self._dirty)

        # Keep all figure-selector labels readable when the theme changes.
        if hasattr(self, "target_btns") and self.target_btns:
            self._refresh_lim_target_button_styles()

        if hasattr(self, "slider_panel"):
            self.slider_panel.setPalette(palette)

            # The bold title uses an explicit foreground color so it stays
            # readable when switching themes.
            title_color = "#f2f2f2" if self.dark_mode else "#111111"
            self.slider_panel.title_label.setStyleSheet(
                f"font-weight: bold; color: {title_color};"
            )

            # Repaint the custom default/YAML slider markers immediately
            # after the palette changes.
            for row in self.slider_panel.rows.values():
                row.slider.update()

    def apply_plot_theme(self):
        self.apply_ui_theme()

        if self.dark_mode:
            fig_bg = "#111111"
            ax_bg = "#181818"
            fg = "white"
            spine = "0.75"
            grid_color = "0.75"
            grid_major_alpha = 0.22
            grid_minor_alpha = 0.12
            legend_face = "#202020"
            legend_edge = "0.65"
        else:
            fig_bg = "white"
            ax_bg = "white"
            fg = "black"
            spine = "black"
            grid_color = "#b0b0b0"
            grid_major_alpha = 0.40
            grid_minor_alpha = 0.40
            legend_face = "white"
            legend_edge = "0.8"

        self.fig.patch.set_facecolor(fig_bg)

        for ax in (self.ax, self.axz, self.axc, self.axcz):
            ax.set_facecolor(ax_bg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            ax.tick_params(axis="x", colors=fg)
            ax.tick_params(axis="y", colors=fg)

            for sp in ax.spines.values():
                sp.set_color(spine)

            ax.grid(True, which="major", color=grid_color, alpha=grid_major_alpha)
            ax.grid(True, which="minor", color=grid_color, alpha=grid_minor_alpha)

        colors, model_colors, thermal_color = self._line_colors_for_theme()

        for k in self.keys:
            self.lines_data[k].set_color(colors[k])
            self.lines_data_z[k].set_color(colors[k])
            self.lines_model[k].set_color(model_colors[k])
            self.lines_model_z[k].set_color(model_colors[k])
            self.lines_classical[k].set_color(colors[k])
            self.lines_classical_z[k].set_color(colors[k])

        self.line_thermal.set_color(thermal_color)
        self.line_thermal_z.set_color(thermal_color)

        self._refresh_legends()

        for ax in (self.ax, self.axc):
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(legend_face)
                leg.get_frame().set_edgecolor(legend_edge)
                for txt in leg.get_texts():
                    txt.set_color(fg)

        self.canvas.draw_idle()
        self.canvas.flush_events()

    def _build_plot_axes(self):
        TOP_GAP = 0.04
        INTERPLOT_GAP = 0.035

        AX_TOP_H = 0.44
        AX_BOT_H = 0.38

        X_LABEL_PAD = 0.4

        ax_top_bottom = 1.0 - TOP_GAP - AX_TOP_H
        ax_bot_bottom = ax_top_bottom - INTERPLOT_GAP - AX_BOT_H

        left = 0.11
        right = 0.06
        total_w = 1.0 - left - right
        gap_mz = 0.02
        main_w = (total_w - gap_mz) * 3.0 / 4.0
        zoom_w = main_w / 3.0
        zoom_left = left + main_w + gap_mz

        self.ax = self.fig.add_axes([left, ax_top_bottom, main_w, AX_TOP_H])
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlim(self.left_xmin0, self.left_xmax0)
        self.ax.set_ylabel(
            "Observed displacement \n ASD [$\mathrm{m}/\sqrt{\mathrm{Hz}}$]",
            fontsize=self.figure_fontsize,
        )
        self.ax.grid(True, which="both", alpha=0.4)
        self.ax.set_ylim(*self.ylim_tl)
        self.ax.tick_params(
            axis="x", which="both", labelbottom=False, labelsize=self.figure_fontsize
        )
        self.ax.tick_params(axis="y", which="both", labelsize=self.figure_fontsize)

        self.axz = self.fig.add_axes([zoom_left, ax_top_bottom, zoom_w, AX_TOP_H])
        self.axz.set_xscale("log")
        self.axz.set_yscale("log")
        self.axz.set_xlim(self.zoom_xmin0, self.zoom_xmax0)
        self.axz.set_ylim(*self.ylim_tr)
        self.axz.grid(True, which="major", alpha=0.40)
        self.axz.grid(True, which="minor", alpha=0.40)
        self.axz.tick_params(
            axis="x", which="both", bottom=False, labelbottom=False,
            labelsize=self.figure_fontsize,
        )
        self.axz.yaxis.set_ticks_position("right")
        self.axz.yaxis.set_label_position("right")
        self.axz.tick_params(
            axis="y", which="both", right=True, labelright=True,
            left=False, labelleft=False, labelsize=self.figure_fontsize,
        )

        self.axc = self.fig.add_axes([left, ax_bot_bottom, main_w, AX_BOT_H])
        self.axc.set_xscale("log")
        self.axc.set_yscale("log")
        self.axc.set_xlim(self.left_xmin0, self.left_xmax0)
        self.axc.set_xlabel(
            "Frequency [Hz]",
            labelpad=X_LABEL_PAD,
            fontsize=self.figure_fontsize,
        )
        self.axc.set_ylabel(
            "Inferred classical \n ASD/ $|\\chi|$ [$\\mathrm{m}/\\sqrt{\\mathrm{Hz}}$]",
            fontsize=self.figure_fontsize,
        )
        self.axc.grid(True, which="both", alpha=0.4)
        self.axc.set_ylim(*self.ylim_bl)
        self.axc.tick_params(axis="x", which="both", labelsize=self.figure_fontsize)
        self.axc.tick_params(axis="y", which="both", labelsize=self.figure_fontsize)

        self.axcz = self.fig.add_axes([zoom_left, ax_bot_bottom, zoom_w, AX_BOT_H])
        self.axcz.set_xscale("log")
        self.axcz.set_yscale("log")
        self.axcz.set_xlim(self.zoom_xmin0, self.zoom_xmax0)
        self.axcz.set_ylim(*self.ylim_br)
        self.axcz.set_xlabel(
            "Frequency [Hz]",
            labelpad=X_LABEL_PAD,
            fontsize=self.figure_fontsize,
        )
        self.axcz.grid(True, which="major", alpha=0.40)
        self.axcz.grid(True, which="minor", alpha=0.40)
        self.axcz.yaxis.set_ticks_position("right")
        self.axcz.yaxis.set_label_position("right")
        self.axcz.tick_params(
            axis="y", which="both", right=True, labelright=True,
            left=False, labelleft=False, labelsize=self.figure_fontsize,
        )

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
            lw=2,
            alpha=1.0,
            ls="-.",
            label="thermal coat",
        )

        (self.line_thermal_z,) = self.axcz.plot(
            self.f_plot,
            np.nan * np.ones_like(self.f_plot),
            color="0.25",
            lw=2,
            alpha=1.0,
            ls="-.",
        )

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

        leg1 = self.ax.legend(
            hh,
            ll,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            fontsize=self.figure_fontsize,
            framealpha=0.85,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.4,
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

        leg2 = self.axc.legend(
            hh,
            ll,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            fontsize=self.figure_fontsize,
            framealpha=0.85,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.2,
            handletextpad=0.45,
            borderaxespad=0.2,
        )

        if self.dark_mode:
            for leg in (leg1, leg2):
                if leg is not None:
                    leg.get_frame().set_facecolor("#202020")
                    leg.get_frame().set_edgecolor("0.65")
                    for txt in leg.get_texts():
                        txt.set_color("white")

    def _axes_for_target(self):
        t = self.lim_target

        if t is None:
            return []

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

        return []

    def _sync_limit_boxes_from_axes(self):
        if self.box_xmin is None:
            return

        axes = self._axes_for_target()

        if len(axes) == 0:
            for box in (self.box_xmin, self.box_xmax, self.box_ymin, self.box_ymax):
                old = box.blockSignals(True)
                box.setText("")
                box.blockSignals(old)
            return

        ax_ref = axes[0]
        xmin, xmax = ax_ref.get_xlim()
        ymin, ymax = ax_ref.get_ylim()

        boxes_vals = [
            (self.box_xmin, f"{xmin:.6g}"),
            (self.box_xmax, f"{xmax:.6g}"),
            (self.box_ymin, f"{ymin:.3e}"),
            (self.box_ymax, f"{ymax:.3e}"),
        ]

        for box, text in boxes_vals:
            old = box.blockSignals(True)
            box.setText(text)
            box.blockSignals(old)

    def _refresh_lim_target_button_styles(self):
        """Refresh selector button colors without changing their geometry."""
        if self.dark_mode:
            selected_bg = "rgb(95, 95, 95)"
            text_color = "#f2f2f2"
        else:
            selected_bg = "rgb(180, 180, 180)"
            text_color = "#111111"

        for k, btn in self.target_btns.items():
            if k == self.lim_target:
                btn.setStyleSheet(
                    f"background-color: {selected_bg}; color: {text_color};"
                )
            else:
                # Do not clear the stylesheet completely: after a button has
                # previously been selected, macOS Qt can retain the old
                # foreground color.  Set only the text color here so the
                # native unselected button background/shape stays untouched.
                btn.setStyleSheet(f"color: {text_color};")

    def _set_lim_target(self, name):
        if name is None:
            self.lim_target = None
        elif str(name) == self.lim_target:
            self.lim_target = None
        else:
            self.lim_target = str(name)

        for k, btn in self.target_btns.items():
            old = btn.blockSignals(True)
            btn.setChecked(k == self.lim_target)
            btn.blockSignals(old)

        self._refresh_lim_target_button_styles()

        self._sync_limit_boxes_from_axes()
        self.canvas.draw_idle()

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

            if self.dark_mode:
                ax.grid(True, which="major", color="0.75", alpha=0.22)
                ax.grid(True, which="minor", color="0.75", alpha=0.12)
            else:
                ax.grid(True, which="major", color="#b0b0b0", alpha=0.40)
                ax.grid(True, which="minor", color="#b0b0b0", alpha=0.40)

            ax.tick_params(axis="x", labelsize=self.figure_fontsize)

        self.axz.tick_params(
            axis="x", which="both", bottom=False, labelbottom=False,
            labelsize=self.figure_fontsize,
        )

    def v(self, name):
        return self.rows[name].value()

    def _make_budget(self, freq):
        import gwinc

        budget = gwinc.load_budget(self.yaml_path, freq, bname="Quantum")
        ifo = budget.ifo
        ifo.Optics.Quadrature.dc = np.pi / 2 + self.LO_ANG_DEG * np.pi / 180
        ifo.Optics.PhotoDetectorEfficiency = self.PDeff
        ifo.Infrastructure.Length = self.arm_length
        return budget

    def _apply_current_params_to_budget(self, budget, key):
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

        return budget

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

    def load_gwinc_yaml(self):
        if yaml is None:
            print("[error] PyYAML is not available, so YAML loading cannot run.")
            return

        inpath_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load GWINC YAML",
            str(Path(self.yaml_path).resolve().parent),
            "YAML files (*.yaml *.yml);;All files (*)",
        )

        if not inpath_str:
            print("GWINC YAML load canceled.")
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

        vals = self._slider_values_from_yaml(cfg)

        if hasattr(self, "slider_panel"):
            self.slider_panel.set_defaults_from_values(vals, set_current=True)

        qdc = self._get_yaml_value(cfg, ("Optics", "Quadrature", "dc"))
        if qdc is not None:
            self.LO_ANG_DEG = float((float(qdc) - np.pi / 2.0) * 180.0 / np.pi)

        pdeff = self._get_yaml_value(cfg, ("Optics", "PhotoDetectorEfficiency"))
        if pdeff is not None:
            self.PDeff = float(pdeff)

        print(f"Loaded GWINC YAML defaults from: {inpath}")
        print(f"Updated {len(vals)} slider/default values.")

        self.recompute()

    def _run_model_quantum(self, key):
        budget = self._make_budget(self.f_plot)
        budget = self._apply_current_params_to_budget(budget, key)

        tr = budget.run()
        asd = np.sqrt(tr.psd)

        if not self._did_unit_check:
            self._did_unit_check = True
            f0 = 100.0
            i0 = int(np.argmin(np.abs(self.f_plot - f0)))
            print(f"[unit check] gwinc sqrt(asd) @ {self.f_plot[i0]:.1f} Hz = {asd[i0]:.3e}")

        return asd

    def _model_as_strain(self, asd_from_gwinc):
        return asd_from_gwinc / self.arm_length

    def reset_defaults(self):
        self.slider_panel.reset_defaults()

    def export_gwinc_params(self):
        if yaml is None:
            print("[error] PyYAML is not available, so YAML export cannot run.")
            return

        if hasattr(self, "slider_panel"):
            self.slider_panel.commit_all_text_values()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"gwinc_fds_fit_params_{timestamp}.yaml"

        outpath_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
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

    def recompute(self):
        if hasattr(self, "slider_panel"):
            self.slider_panel.commit_all_text_values()

        self._dirty = False

        for k in self.keys:
            self.lines_model[k].set_linestyle(":")
            self.lines_model[k].set_alpha(1.0)
            self.lines_model_z[k].set_linestyle(":")
            self.lines_model_z[k].set_alpha(1.0)

        self.canvas.draw_idle()
        self.canvas.flush_events()

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

            if self.dark_mode:
                self.apply_plot_theme()

            self.canvas.draw_idle()
            self.canvas.flush_events()

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

        self.canvas.draw_idle()
        self.canvas.flush_events()

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

        self.canvas.draw_idle()
        self.canvas.flush_events()

    def apply_limits(self):
        xmin = safe_float(self.box_xmin.text())
        xmax = safe_float(self.box_xmax.text())
        ymin = safe_float(self.box_ymin.text())
        ymax = safe_float(self.box_ymax.text())

        tgt_axes = self._axes_for_target()

        if xmin is not None and xmax is not None and xmin > 0 and xmax > xmin:
            for ax in tgt_axes:
                ax.set_xlim(xmin, xmax)

        if ymin is not None and ymax is not None and ymin > 0 and ymax > ymin:
            for ax in tgt_axes:
                ax.set_ylim(ymin, ymax)

        if any(ax in (self.axz, self.axcz) for ax in tgt_axes):
            self._update_zoom_xticks()

        if self.dark_mode:
            self.apply_plot_theme()

        self.canvas.draw_idle()
        self.canvas.flush_events()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        qapp = QtWidgets.QApplication([])

    HERE = Path(__file__).resolve().parent
    os.chdir(HERE)

    yaml_path = HERE / "april9.yaml"
    data_mat = HERE / "calibrated_darm_err_with_resonance.mat"
    oltf_res_mat = HERE / "unwrapped_oltf_with_resonance.mat"
    oltf_nores_mat = HERE / "unwrapped_oltf_no_resonance.mat"

    window = App(
        yaml_path=str(yaml_path),
        data_mat=data_mat,
        oltf_res_mat=oltf_res_mat,
        oltf_nores_mat=oltf_nores_mat,
        keys=("no_sqz", "fis", "fds"),
        fmin=20,
        fmax=7000,
    )

    window.show()

    try:
        qapp.exec()
    except AttributeError:
        qapp.exec_()