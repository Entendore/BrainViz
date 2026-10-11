import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from nilearn import datasets, plotting, surface
from scipy.interpolate import splprep, splev
from scipy.signal import spectrogram

# ==========================================
# 1. DATA FETCHING & SIMULATION
# ==========================================
print("Initializing PySide6 Neuro-Tabbed Desktop App...")
print("Fetching data...")
mni_brain = datasets.load_mni152_template()
motor_task = datasets.fetch_localizer_motor_task()
stat_map = motor_task.cmaps[0]
coords = datasets.fetch_coords_power_2011()
n_nodes = len(coords.rois)
fsaverage = datasets.fetch_surf_fsaverage()
destrieux = datasets.fetch_atlas_surf_destrieux()

print("Processing 3D surfaces and simulating data...")
texture_right = surface.vol_to_surf(stat_map, fsaverage.pial_right)
texture_left = surface.vol_to_surf(stat_map, fsaverage.pial_left)

# Simulate Connectome & Yeo Colors
yeo_colors = ['#782282', '#4682B4', '#00760E', '#C43A04', '#7BAF2C', '#D49137', '#916a4f']
node_networks = np.random.randint(0, 7, n_nodes)
node_colors = [yeo_colors[i] for i in node_networks]
corr_matrix = np.random.randn(n_nodes, n_nodes) * 0.15
block_size = n_nodes // 7
for i in range(7):
    start, end = i * block_size, (i + 1) * block_size
    corr_matrix[start:end, start:end] += 0.8
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 1)
matrix_thresholded = corr_matrix.copy()
matrix_thresholded[np.abs(corr_matrix) < 0.4] = 0

# Simulate DTI Tracts
dti_tubes = []
for _ in range(25):
    i, j = np.random.randint(0, n_nodes, 2)
    if i != j:
        p0 = np.array(coords.rois[i]); p1 = np.array(coords.rois[j])
        mid_point = (p0 + p1) / 2 + np.random.normal(0, 15, 3)
        points = np.array([p0, mid_point, p1])
        tck, u = splprep(points.T, u=None, k=2, s=0) 
        u_new = np.linspace(u.min(), u.max(), 50)
        tube = np.array(splev(u_new, tck)).T
        dti_tubes.append((tube, yeo_colors[node_networks[i]]))

# Simulate ECoG Grid
ecog_coords = []
for x in np.linspace(35, 50, 8):
    for y in np.linspace(-30, -10, 8):
        z = np.sqrt(max(1, 45**2 - x**2 - y**2)) + np.random.normal(0, 1.5)
        ecog_coords.append([x, y, z])
ecog_coords = np.array(ecog_coords)

# Simulate Time-Series Data
tr = 2.0; n_scans = 200
time_points = np.arange(n_scans) * tr
task_blocks = np.zeros(n_scans)
for i in range(0, n_scans, 20):
    if (i // 20) % 2 != 0: task_blocks[i:i+20] = 1
bold_signal = np.convolve(task_blocks, np.ones(5)/5, mode='same') * 2.5 + 100
eeg_fast = task_blocks * 2 + np.random.normal(0, 0.2, n_scans)

# Simulate Carpet Plot
carpet = np.random.randn(100, 200) * 0.5
for i in range(0, 200, 40): carpet[0:30, i:i+20] += 2.0
carpet[70:100, :] += np.sin(np.linspace(0, 10*np.pi, 200)) * 1.0


# ==========================================
# 2. PYSIDE6 GUI & TAB SETUP
# ==========================================
class NeuroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Intelligence & Surgical Mapping Suite (PySide6)")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: #020205;")

        # Create Main Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 0px; }
            QTabBar::tab { background: #0a0a12; color: gray; padding: 15px 30px; font-size: 16px; font-weight: bold; }
            QTabBar::tab:selected { background: #12121f; color: #00f2ff; border-bottom: 3px solid #00f2ff; }
        """)
        self.setCentralWidget(self.tabs)

        # Build Tabs
        self.create_surgical_tab()
        self.create_dynamics_tab()
        self.create_connectomics_tab()

    def add_canvas_to_tab(self, fig):
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: transparent;")
        return canvas

    # --- TAB 1: 3D Surgical Core ---
    def create_surgical_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        fig = plt.figure(figsize=(16, 9), facecolor='#020205')
        gs = GridSpec(1, 3, figure=fig, wspace=0.05)

        # Left: Pial Surface Anatomy
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        plotting.plot_surf_roi(fsaverage.pial_left, destrieux.map_left, hemi='left', view='lateral',
                               title="Anatomy: Left Hemisphere", bg_map=fsaverage.sulc_left, 
                               axes=ax1, cmap='nipy_spectral', black_bg=True, darkness=0.2)
        ax1.set_axis_off()

        # Center: Surgical Map with ECoG
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        plotting.plot_surf_stat_map(fsaverage.pial_right, texture_right, hemi='right', view='lateral',
                                    title="Surgery: ECoG Grid & Activity", threshold=3.0, cmap='inferno', 
                                    bg_map=fsaverage.sulc_right, axes=ax2, black_bg=True, darkness=0.1, alpha=0.9)
        ax2.scatter(ecog_coords[:,0], ecog_coords[:,1], ecog_coords[:,2], 
                    color='#FFD700', s=80, edgecolors='black', linewidths=1, depthshade=True)
        ax2.set_axis_off()

        # Right: DTI Tracts & Lesion
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.set_title("Pathology: DTI Tracts & Lesion", color='#00f2ff', fontsize=14, pad=10)
        plotting.plot_glass_brain(mni_brain, axes=ax3, black_bg=True, display_mode='lzr', alpha=0.1, threshold=0)
        for tube, color in dti_tubes:
            ax3.plot(tube[:,0], tube[:,1], tube[:,2], color=color, alpha=0.6, linewidth=1)
        # Lesion
        lesion_center = [40, -10, -20]
        ax3.scatter(lesion_center[0], lesion_center[1], lesion_center[2], color='red', s=500, alpha=0.4)
        ax3.scatter(lesion_center[0], lesion_center[1], lesion_center[2], color='darkred', s=200, alpha=0.9)
        ax3.set_xlim(-50, 50); ax3.set_ylim(-60, 50); ax3.set_zlim(-40, 50)
        ax3.set_axis_off()

        layout.addWidget(self.add_canvas_to_tab(fig))
        self.tabs.addTab(tab, "🧠 3D Surgical Core")

    # --- TAB 2: Multimodal Dynamics ---
    def create_dynamics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        fig = plt.figure(figsize=(16, 9), facecolor='#020205')
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

        # Top Left: BOLD & EEG Dual Axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_points, bold_signal, color='#00f2ff', linewidth=2, label='fMRI BOLD')
        ax1_2 = ax1.twinx()
        ax1_2.plot(time_points, eeg_fast * 5 + 100, color='#ff0055', linewidth=1.5, linestyle='--', label='EEG')
        ax1.set_title("Neurovascular Coupling", color='white', fontsize=14)
        ax1.tick_params(colors='gray'); ax1_2.tick_params(colors='gray')
        ax1.grid(color='gray', linestyle='--', alpha=0.2)

        # Top Right: EEG Spectrogram
        ax2 = fig.add_subplot(gs[0, 1])
        fs_eeg = 1000; t_eeg = np.linspace(0, 20, fs_eeg * 20)
        eeg_signal = np.sin(2 * np.pi * 10 * t_eeg)
        for start_t in range(4, 20, 8):
            mask = (t_eeg >= start_t) & (t_eeg < start_t + 4)
            eeg_signal[mask] += np.sin(2 * np.pi * 25 * t_eeg[mask]) * 1.5
        eeg_signal += np.random.normal(0, 0.5, len(t_eeg))
        f, t, Sxx = spectrogram(eeg_signal, fs_eeg, nperseg=512, noverlap=256)
        ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-6), cmap='magma', shading='gouraud')
        ax2.set_ylim(1, 40); ax2.set_title("EEG Spectrogram", color='white', fontsize=14)
        ax2.tick_params(colors='gray')

        # Bottom Left: fMRI Carpet Plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow((carpet - carpet.mean(axis=1, keepdims=True)) / carpet.std(axis=1, keepdims=True), 
                   aspect='auto', cmap='cold_hot', vmin=-3, vmax=3, interpolation='nearest')
        ax3.set_title("fMRI Carpet Plot", color='white', fontsize=14)
        ax3.tick_params(colors='gray')

        # Bottom Right: Dynamic Matrices
        gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.1)
        states = [np.random.randn(50,50)*0.2, np.random.randn(50,50)*0.2, np.random.randn(50,50)*0.2]
        states[1][0:10, 0:10] = 0.8; states[2][0:15, 0:15] = 0.9
        titles = ["Rest", "Onset", "Peak"]
        colors = ['gray', 'yellow', 'lime']
        for i, ax in enumerate([fig.add_subplot(gs_inner[i]) for i in range(3)]):
            ax.imshow(states[i], cmap='viridis', vmin=-0.5, vmax=1, aspect='auto')
            ax.set_title(titles[i], color=colors[i], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        layout.addWidget(self.add_canvas_to_tab(fig))
        self.tabs.addTab(tab, "📈 Multimodal Dynamics")

    # --- TAB 3: Connectomics ---
    def create_connectomics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        fig = plt.figure(figsize=(16, 9), facecolor='#020205')
        gs = GridSpec(1, 2, figure=fig, wspace=0.05)

        # Left: 3D Connectome
        ax1 = fig.add_subplot(gs[0, 0])
        plotting.plot_connectome(matrix_thresholded, coords.rois, title="3D Yeo Network Connectome",
                                 axes=ax1, node_color=node_colors, edge_color='w', node_size=30, 
                                 black_bg=True, edge_threshold='85%')
        
        # Right: Correlation Matrix Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(matrix_thresholded, cmap='magma', vmin=0, vmax=1, aspect='auto')
        ax2.set_title("Functional Connectivity Matrix", color='white', fontsize=14)
        ax2.tick_params(colors='gray')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        layout.addWidget(self.add_canvas_to_tab(fig))
        self.tabs.addTab(tab, "🌐 Connectomics")

# ==========================================
# 3. RUN THE PYSIDE6 APPLICATION
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Force dark mode palette for PySide6 UI elements
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(2, 2, 5))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = NeuroApp()
    window.show()
    sys.exit(app.exec()) # PySide6 uses exec() instead of exec_()