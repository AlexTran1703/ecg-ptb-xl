import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QListWidget, QToolButton,
    QStackedLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QColor

dataset_colors = {
    'train': QColor("#d0f5d8"),  # Light green
    'test': QColor("#f5d0d0"),   # Light red
    'valid': QColor("#d0d5f5")   # Light blue
}

# File paths
path_load = '../data_source_sub'

dataset_files = {
    'train': {
        'signal': os.path.join(path_load, 'Y_train.npy'),
        'super': os.path.join(path_load, 'Z_train.csv'),
        'sub': os.path.join(path_load, 'T_train.csv')
    },
    'test': {
        'signal': os.path.join(path_load, 'Y_test.npy'),
        'super': os.path.join(path_load, 'Z_test.csv'),
        'sub': os.path.join(path_load, 'T_test.csv')
    },
    'valid': {
        'signal': os.path.join(path_load, 'Y_valid.npy'),
        'super': os.path.join(path_load, 'Z_valid.csv'),
        'sub': os.path.join(path_load, 'T_valid.csv')
    }
}

superclass_dict = {
    'NORM': ['NORM'],
    'CD': ['LAFB/LPFB', 'IRBBB', 'ILBBB', 'CLBBB', 'CRBBB', '_AVB', 'IVCB'],
    'HYP': ['LVH', 'RVH', 'LAO/LAE', 'RAO/RAE', 'SEHYP'],
    'MI': ['AMI', 'IMI', 'LMI', 'PMI', 'ISCA', 'ISCI'],
    'STTC': ['ISC_', 'STTC', 'NST_']
}

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead_dict = {i: lead for i, lead in enumerate(lead_names)}

class ECGViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PTB-XL ECG Multi-Lead Viewer")
        self.showMaximized()

        # Load datasets
        self.datasets = {name: np.load(files['signal']) for name, files in dataset_files.items()}
        self.super_classes = {name: pd.read_csv(files['super']) for name, files in dataset_files.items()}
        self.sub_classes = {name: pd.read_csv(files['sub']) for name, files in dataset_files.items()}

        self.super_label_names = list(self.super_classes['train'].columns[1:])
        self.sub_label_names = list(self.sub_classes['train'].columns[1:])
        self.super_label_map = {label: i for i, label in enumerate(self.super_label_names)}
        self.sub_label_map = {label: i for i, label in enumerate(self.sub_label_names)}

        self.current_dataset_name = 'train'
        self.current_data = self.datasets[self.current_dataset_name]

        self.selected_super_label = None
        self.selected_sub_labels = []
        self.available_samples = []

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.stacked_layout = QStackedLayout()

        # Page selector
        self.page_selector = QComboBox()
        self.page_selector.addItems(["ECG Viewer", "Biostatistics"])
        self.page_selector.currentIndexChanged.connect(self.switch_page)
        self.layout.addWidget(self.page_selector)

        self.viewer_widget = QWidget()
        self.init_viewer_page()
        self.stacked_layout.addWidget(self.viewer_widget)

        self.stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        stats_layout.addWidget(self.stats_table)
        self.stats_widget.setLayout(stats_layout)
        self.stacked_layout.addWidget(self.stats_widget)

        self.layout.addLayout(self.stacked_layout)
        self.setLayout(self.layout)
        self.update_biostatistics()

    def switch_page(self, index):
        self.stacked_layout.setCurrentIndex(index)

    def init_viewer_page(self):
        layout = QVBoxLayout()
        self.viewer_widget.setLayout(layout)

        control_layout = QHBoxLayout()
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(['train', 'test', 'valid'])
        self.dataset_selector.currentTextChanged.connect(self.change_dataset)

        self.index_spinner = QSpinBox()
        self.index_spinner.setMinimum(0)
        self.index_spinner.setMaximum(len(self.current_data) - 1)

        self.plot_button = QPushButton("Plot Selected Labels")
        self.plot_button.clicked.connect(self.plot_selected_labels)

        self.prev_button = QToolButton()
        self.prev_button.setText("←")
        self.prev_button.clicked.connect(self.go_to_previous_sample)

        self.next_button = QToolButton()
        self.next_button.setText("→")
        self.next_button.clicked.connect(self.go_to_next_sample)

        control_layout.addWidget(QLabel("Dataset:"))
        control_layout.addWidget(self.dataset_selector)
        control_layout.addWidget(QLabel("Sample Index:"))
        control_layout.addWidget(self.index_spinner)
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.plot_button)
        layout.addLayout(control_layout)

        layout.addWidget(QLabel("Select Super Class:"))
        self.super_label_selector = QComboBox()
        self.super_label_selector.addItems(self.super_label_names)
        self.super_label_selector.currentTextChanged.connect(self.update_super_label)
        layout.addWidget(self.super_label_selector)

        layout.addWidget(QLabel("Select Sub Classes:"))
        self.sub_label_selector = QListWidget()
        self.sub_label_selector.addItems(self.sub_label_names)
        self.sub_label_selector.setSelectionMode(QListWidget.MultiSelection)
        self.sub_label_selector.itemSelectionChanged.connect(self.update_sub_labels)
        layout.addWidget(self.sub_label_selector)

        layout.addWidget(QLabel("Available Samples:"))
        self.sample_selector = QComboBox()
        self.sample_selector.addItem("Select Sample")
        self.sample_selector.currentIndexChanged.connect(self.update_index_from_sample)
        layout.addWidget(self.sample_selector)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update_biostatistics(self):
        headers = ["Class Label"] + [f"{ds.upper()} (Super)" for ds in dataset_files] + [f"{ds.upper()} (Sub)" for ds in dataset_files]
        class_rows = []
        for super_label, sub_labels in superclass_dict.items():
            class_rows.append((super_label, None))
            for sub in sub_labels:
                class_rows.append((super_label, sub))

        self.stats_table.setRowCount(len(class_rows))
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)

        for row_idx, (super_label, sub_label) in enumerate(class_rows):
            label_text = f"[{super_label}]" if sub_label is None else f"   ↳ {sub_label}"
            self.stats_table.setItem(row_idx, 0, QTableWidgetItem(label_text))

            for col_idx, ds in enumerate(dataset_files):
                super_df = self.super_classes[ds]
                sub_df = self.sub_classes[ds]

                if sub_label is None:
                    super_count = super_df[super_label].sum() if super_label in super_df.columns else 0
                    sub_count = sum(sub_df[sub].sum() for sub in superclass_dict[super_label] if sub in sub_df.columns)
                else:
                    super_count = ""
                    sub_count = sub_df[sub_label].sum() if sub_label in sub_df.columns else 0
                # Set super class count
                item_super = QTableWidgetItem(str(super_count))
                item_super.setBackground(dataset_colors[ds])
                self.stats_table.setItem(row_idx, 1 + col_idx, item_super)

                # Set sub class count
                item_sub = QTableWidgetItem(str(sub_count))
                item_sub.setBackground(dataset_colors[ds])
                self.stats_table.setItem(row_idx, 1 + len(dataset_files) + col_idx, item_sub)
                
                #self.stats_table.setItem(row_idx, 1 + len(dataset_files) + col_idx, item_sub)
                self.stats_table.setItem(row_idx, 1 + col_idx, QTableWidgetItem(str(super_count)))
                #self.stats_table.setItem(row_idx, 1 + len(dataset_files) + col_idx, QTableWidgetItem(str(sub_count)))

        self.stats_table.resizeColumnsToContents()
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # Reuse your previous viewer code: change_dataset, update_super_label, update_sub_labels, etc.
    def change_dataset(self, name):
        self.current_dataset_name = name
        self.current_data = self.datasets[name]
        self.index_spinner.setMaximum(len(self.current_data) - 1)
        self.update_available_samples()

    def update_super_label(self, label):
        self.selected_super_label = label
        self.update_available_samples()

    def update_sub_labels(self):
        self.selected_sub_labels = [item.text() for item in self.sub_label_selector.selectedItems()]
        self.update_available_samples()

    def update_available_samples(self):
        if not self.selected_super_label:
            return

        super_idx = self.super_label_map[self.selected_super_label]
        sub_indices = [self.sub_label_map[sub] for sub in self.selected_sub_labels]

        available = []
        super_df = self.super_classes[self.current_dataset_name]
        sub_df = self.sub_classes[self.current_dataset_name]

        for i in range(len(super_df)):
            if super_df.iloc[i, super_idx + 1] == 1:
                if sub_indices:
                    if any(sub_df.iloc[i, sub_idx + 1] == 1 for sub_idx in sub_indices):
                        available.append(i)
                else:
                    available.append(i)

        self.sample_selector.clear()
        if available:
            self.sample_selector.addItem("Select Sample")
            for i in available:
                self.sample_selector.addItem(str(i))
            self.available_samples = available
        else:
            self.sample_selector.addItem("No samples available")
            self.available_samples = []

    def update_index_from_sample(self):
        idx = self.sample_selector.currentIndex() - 1
        if 0 <= idx < len(self.available_samples):
            self.index_spinner.setValue(self.available_samples[idx])
            self.plot_selected_labels()

    def decode_multilabel(self, row, names):
        return [names[i] for i, val in enumerate(row.iloc[1:].values) if val == 1]

    def plot_selected_labels(self):
        idx = self.index_spinner.value()
        data = self.current_data[idx]

        super_row = self.super_classes[self.current_dataset_name].iloc[idx]
        sub_row = self.sub_classes[self.current_dataset_name].iloc[idx]

        super_labels = self.decode_multilabel(super_row, self.super_label_names)
        sub_labels = self.decode_multilabel(sub_row, self.sub_label_names)

        self.figure.clear()
        T, C = data.shape
        for i in range(C):
            rows, cols = 4, 3
            row = i % rows
            col = i // rows
            ax = self.figure.add_subplot(rows, cols, row * cols + col + 1)
            ax.plot(data[:, i], linewidth=0.8)
            ax.set_title(f"Lead {lead_dict[i]}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        title = f"True Super: {', '.join(super_labels)} | True Sub: {', '.join(sub_labels)}"
        self.figure.suptitle(title, fontsize=16)
        self.figure.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()

    def go_to_previous_sample(self):
        self.update_available_samples()
        current_idx = self.index_spinner.value()
        try:
            current_pos = self.available_samples.index(current_idx)
            if current_pos > 0:
                self.index_spinner.setValue(self.available_samples[current_pos - 1])
                self.plot_selected_labels()
        except ValueError:
            pass

    def go_to_next_sample(self):
        self.update_available_samples()
        current_idx = self.index_spinner.value()
        try:
            current_pos = self.available_samples.index(current_idx)
            if current_pos < len(self.available_samples) - 1:
                self.index_spinner.setValue(self.available_samples[current_pos + 1])
                self.plot_selected_labels()
        except ValueError:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ECGViewer()
    viewer.show()
    sys.exit(app.exec_())
