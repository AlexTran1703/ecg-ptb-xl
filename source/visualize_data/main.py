import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QListWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead_dict = {i: lead for i, lead in enumerate(lead_names)}

class ECGViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Load datasets
        self.datasets = {name: np.load(files['signal']) for name, files in dataset_files.items()}
        self.super_classes = {name: pd.read_csv(files['super']) for name, files in dataset_files.items()}
        self.sub_classes = {name: pd.read_csv(files['sub']) for name, files in dataset_files.items()}

        # Extract label names
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
        self.setWindowTitle("PTB-XL ECG Multi-Lead Viewer")
        self.showMaximized()
        layout = QVBoxLayout()

        # Control layout
        control_layout = QHBoxLayout()
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(['train', 'test', 'valid'])
        self.dataset_selector.currentTextChanged.connect(self.change_dataset)

        self.index_spinner = QSpinBox()
        self.index_spinner.setMinimum(0)
        self.index_spinner.setMaximum(len(self.current_data) - 1)

        self.plot_button = QPushButton("Plot Selected Labels")
        self.plot_button.clicked.connect(self.plot_selected_labels)

        control_layout.addWidget(QLabel("Dataset:"))
        control_layout.addWidget(self.dataset_selector)
        control_layout.addWidget(QLabel("Sample Index:"))
        control_layout.addWidget(self.index_spinner)
        control_layout.addWidget(self.plot_button)
        layout.addLayout(control_layout)

        # Super label
        layout.addWidget(QLabel("Select Super Class:"))
        self.super_label_selector = QComboBox()
        self.super_label_selector.addItems(self.super_label_names)
        self.super_label_selector.currentTextChanged.connect(self.update_super_label)
        layout.addWidget(self.super_label_selector)

        # Sub labels
        layout.addWidget(QLabel("Select Sub Classes:"))
        self.sub_label_selector = QListWidget()
        self.sub_label_selector.addItems(self.sub_label_names)
        self.sub_label_selector.setSelectionMode(QListWidget.MultiSelection)
        self.sub_label_selector.itemSelectionChanged.connect(self.update_sub_labels)
        layout.addWidget(self.sub_label_selector)

        # Sample selector
        layout.addWidget(QLabel("Available Samples:"))
        self.sample_selector = QComboBox()
        self.sample_selector.addItem("Select Sample")
        self.sample_selector.currentIndexChanged.connect(self.update_index_from_sample)
        layout.addWidget(self.sample_selector)

        # Plot area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

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

    def decode_multilabel(self, row, names):
        return [names[i] for i, val in enumerate(row.iloc[1:].values) if val == 1]

    def plot_selected_labels(self):
        idx = self.index_spinner.value()
        data = self.current_data[idx]

        super_row = self.super_classes[self.current_dataset_name].iloc[idx]
        sub_row = self.sub_classes[self.current_dataset_name].iloc[idx]

        super_labels = self.decode_multilabel(super_row, self.super_label_names)
        sub_labels = self.decode_multilabel(sub_row, self.sub_label_names)

        if self.selected_super_label not in super_labels:
            return
        if self.selected_sub_labels and not any(lbl in sub_labels for lbl in self.selected_sub_labels):
            return

        self.figure.clear()
        T, C = data.shape
        for i in range(C):
            ax = self.figure.add_subplot(4, 3, i + 1)
            ax.plot(data[:, i], linewidth=0.8)
            ax.set_title(f"Lead {lead_dict[i]}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        title = f"True Super: {', '.join(super_labels)} | True Sub: {', '.join(sub_labels)}"
        self.figure.suptitle(title, fontsize=16)
        self.figure.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ECGViewer()
    viewer.show()
    sys.exit(app.exec_())
