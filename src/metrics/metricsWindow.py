import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from queue import Empty
from utils.plot import *
from utils.tkinter import *
import numpy as np
import pandas as pd
import re

sns.set_theme()

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)    
plt.rc('axes', labelsize=MEDIUM_SIZE)   
plt.rc('xtick', labelsize=SMALL_SIZE)   
plt.rc('ytick', labelsize=SMALL_SIZE)   
plt.rc('legend', fontsize=MEDIUM_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE) 


class MetricsWindow():

    def __init__(self, genes, actions, queue, event_close, time_freq = 1000) -> None:
        self.genes = genes
        self.n_genes = len(genes)
        self.actions = actions
        self.n_actions = len(actions)
        self.queue = queue
        self.event_close = event_close
        self.time_freq = time_freq
        self.labels = []

    def run(self):
        self.root = tk.Tk()
        self.root.title("Metrics")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_top_bar()
        self._build_main()
        
        self.update()
        self.root.mainloop()
        
    def _build_top_bar(self):
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", padx=6, pady=6)
        self.step_label = tk.Label(top, text="STEP: 0", font=("Arial", 16))
        self.step_label.pack(side="left", padx=(0, 12))
        self.step_per_s_label = tk.Label(top, text="STEP PER S: 0", font=("Arial", 16))
        self.step_per_s_label.pack(side="left", padx=(0, 12))
        
    def _build_main(self):
        main = tk.Frame(self.root)
        main.pack(side="top", fill="both", expand=True)
        self._build_left_column(main)
        self._build_mid_column(main)
        self._build_right_column(main)
        
    def _build_left_column(self, parent):
        left = tk.Frame(parent)
        left.pack(side="left", fill="y", expand=True, padx=4, pady=4)
        tk.Label(left, text="General Stats", font=("Arial", 14, "bold")).pack()

        # --- General Plots ---
        plots = tk.Frame(left)
        plots.pack(side="top", fill="both", expand=True)
        self.left_fig1, self.left_axs1 = plt.subplots(1, 3, figsize=(6, 2))
        self.left_canvas1 = self._add_canvas(plots, self.left_fig1)
        
        # --- Species Plot ---
        # control
        controls = tk.Frame(left)
        controls.pack(fill="x")
        row_pca = tk.Frame(controls)
        row_pca.pack(anchor="w")
        tk.Label(row_pca, text="Species plot:").pack(side="left")
        self.species_plot_selector = MenuSelector(row_pca, ["PCA (brain)", "PCA (genome)", "PCA (both)", "Species Density"], default="PCA (both)")
        self.species_plot_selector.pack(side="left")
        # plots
        plots = tk.Frame(left)
        plots.pack(side="top", fill="both", expand=True)
        self.left_fig2 = plt.figure(figsize=(6, 6))
        self.left_ax2 = self.left_fig2.add_subplot(111)
        self.left_canvas2 = self._add_canvas(plots, self.left_fig2, expand=True)
        
    def _build_mid_column(self, parent):
        mid = tk.Frame(parent)
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        tk.Label(mid, text="Genes", font=("Arial", 14, "bold")).pack(pady=(10, 0))
        self.mid_fig1, self.mid_axs1 = plt.subplots(2, self.n_genes, figsize=(6, 3))
        self.mid_canvas1 = self._add_canvas(mid, self.mid_fig1)

        tk.Label(mid, text="Brain", font=("Arial", 14, "bold")).pack(pady=(10, 0))
        self.mid_fig2 = plt.figure(figsize=(6, 2))
        self.mid_ax2 = self.mid_fig2.add_subplot(111)
        self.mid_canvas2 = self._add_canvas(mid, self.mid_fig2)

        # --- Controls ---
        row_brain = tk.Frame(mid)
        row_brain.pack()

        tk.Label(row_brain, text="selected action:").pack(side="left")
        self.brainaction_selector = MenuSelector(row_brain, self.actions, default=self.actions[0])
        self.brainaction_selector.pack(side="left")

        # --- Plots ---
        self.mid_fig3, self.mid_axs3 = plt.subplots(1, 2, figsize=(6, 3))
        self.mid_canvas3 = self._add_canvas(mid, self.mid_fig3)
        
    def _build_right_column(self, parent):
        right = tk.Frame(parent)
        right.pack(side="left", fill="y", padx=4, pady=4)

        tk.Label(right, text="Specie details", font=("Arial", 14, "bold")).pack()

        row_species = tk.Frame(right)
        row_species.pack()

        tk.Label(row_species, text="selected species:").pack(side="left")
        self.species_selector = MenuSelector(row_species, [])
        self.species_selector.pack(side="left")
        
        ### Genome
        tk.Label(right, text="Genome", font=("Arial", 10, "bold")).pack()

        self.right_fig1 = plt.figure(figsize=(5, 2))
        self.right_ax1 = self.right_fig1.add_subplot(111)
        self.right_canvas1 = self._add_canvas(right, self.right_fig1, expand=True)

        self.right_fig2 = plt.figure(figsize=(5, 2))
        self.right_ax2 = self.right_fig2.add_subplot(111)
        self.right_canvas2 = self._add_canvas(right, self.right_fig2, expand=True)
        
        ### Brain
        tk.Label(right, text="Brain", font=("Arial", 10, "bold")).pack()
        
        # brain action control
        row_brain = tk.Frame(right)
        row_brain.pack()
        tk.Label(row_brain, text="selected action:").pack(side="left")
        self.species_brainaction_selector = MenuSelector(row_brain, self.actions, default=self.actions[0])
        self.species_brainaction_selector.pack(side="left")
        
        self.right_fig3 = plt.figure(figsize=(5, 2))
        self.right_ax3 = self.right_fig3.add_subplot(111)
        self.right_canvas3 = self._add_canvas(right, self.right_fig3, expand=True)
        
        self.right_fig4 = plt.figure(figsize=(5, 2))
        self.right_ax4 = self.right_fig4.add_subplot(111)
        self.right_canvas4 = self._add_canvas(right, self.right_fig4, expand=True)
        
    def _add_canvas(self, parent, fig, expand=False):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=expand)
        canvas.get_tk_widget().configure(takefocus=0)
        return canvas


    def update(self):
        try:
            metrics = self.queue.get_nowait()
        except Empty:
            metrics = None
        if metrics is not None:
            try:
                self.update_buttons(metrics)
                self.update_plots(metrics)
                self.update_labels(metrics)  
            except Exception as e:
                print(e)
     
        self.left_canvas1.draw_idle()
        self.left_canvas2.draw_idle()
        self.mid_canvas1.draw_idle()
        self.mid_canvas2.draw_idle()
        self.mid_canvas3.draw_idle()
        self.right_canvas1.draw_idle()
        self.right_canvas2.draw_idle()
        self.right_canvas3.draw_idle()
        self.right_canvas4.draw_idle()
        self.root.after(self.time_freq, self.update)
   
        
    def update_labels(self, metrics):
        self.step_label.config(text=f"STEP: {metrics.get("n_step", 0)}")
        self.step_per_s_label.config(text=f"FPS (cur|avg): {metrics.get("fps_cur", 0):.1f} | {metrics.get("fps_avg", 0):.1f}")
        
        
    def update_buttons(self, metrics):
        self.species_selector.set_values(metrics['species'])
        self.brainaction_selector.set_values(self.actions)
      
        
    def update_plots(self, metrics):
        steps = metrics['steps']
        # == Left ==
        
        # general stats
            # population
        line_plot(self.left_axs1[0], steps, metrics["population"], 'Population', ylim=0)
            # births
        line_plot(self.left_axs1[1], steps, metrics["births"], 'Births')
            # deaths
        multiline_plot(self.left_axs1[2], steps, metrics["deaths"].values(), metrics["deaths"].keys(), 'Deaths')
        
        # species plot
        plot_type = self.species_plot_selector.get()
        if plot_type.split()[0] == 'PCA' and metrics["species_pca_data"] is not None:
            m = re.search(r'\(([^)]+)\)', plot_type)
            if m:
                pca_attr = m.group(1)
                scatter_plot(self.left_ax2, **metrics["species_pca_data"][pca_attr], title='Species scatterplot')
        elif plot_type == 'Species Density' and metrics['species_density_data'] is not None:
            species_density_df = pd.DataFrame(metrics['species_density_data'])
            density_plot(self.left_ax2 , species_density_df, 'step', 'count', 'species', "Species proportions over time")
        elif plot_type == 'dendrogram':
            pass
        
        # == Middle ==
        # genes
        line_multiplot(self.mid_axs1[0], steps, metrics[f"genes_means"].values(), [f'{g} mean' for g in metrics[f"genes_means"].keys()])
        line_multiplot(self.mid_axs1[1], steps, metrics[f"genes_cvs"].values(), [f'{g} CV' for g in metrics[f"genes_cvs"].keys()])
        # brain 
        actions_density_df = pd.DataFrame(metrics['actions_density_data'])
        density_plot(self.mid_ax2, actions_density_df, 'step', 'count', 'action', "Actions density")
        brain_action = self.brainaction_selector.get()
        action_means = metrics['brains_means'][brain_action]
        action_stds = metrics['brains_stds'][brain_action]
        multiline_plot(self.mid_axs3[0], steps, action_means.values(), action_means.keys(), f'{brain_action} mean')
        multiline_plot(self.mid_axs3[1], steps, action_stds.values(), action_stds.keys(), f'{brain_action} std')
        
        # == Right ==
        selected_species = self.species_selector.get()
        x = np.arange(self.n_genes)
        species = metrics["species"]
        # species genes mean
        genes_mean = metrics["species_genes_mean"]
        min_g_mean = min(min(genes_mean[s]) for s in species)
        max_g_mean = max(max(genes_mean[s]) for s in species)
        bar_plot(self.right_ax1, x, genes_mean[selected_species], selected_species, self.genes, 'Genes means', ymin=min_g_mean, ymax= max_g_mean)
        # species genes cv
        genes_cv = metrics["species_genes_cv"]
        min_g_cv = min(min(genes_cv[s]) for s in species)
        max_g_cv = max(max(genes_cv[s]) for s in species)
        bar_plot(self.right_ax2, x, genes_cv[selected_species], selected_species, self.genes, 'Genes CVs', ymin=min_g_cv, ymax= max_g_cv)
        # brain
        selected_action = self.species_brainaction_selector.get()
        brains_mean = metrics["species_brains_mean"]
        brains_std = metrics["species_brains_std"]
        labels = brains_mean[selected_species][selected_action].keys()
        x = np.arange(len(labels))
        min_b_mean = min(min(brains_mean[s][selected_action].values()) for s in species)
        max_b_mean = max(max(brains_mean[s][selected_action].values()) for s in species)
        bar_plot(self.right_ax3, x, metrics["species_brains_mean"][selected_species][selected_action].values(), selected_species, labels, 'Brain means', ymin=min_b_mean, ymax= max_b_mean)
        min_b_std = min(min(brains_std[s][selected_action].values()) for s in species)
        max_b_std = max(max(brains_std[s][selected_action].values()) for s in species)
        bar_plot(self.right_ax4, x, metrics["species_brains_std"][selected_species][selected_action].values(), selected_species, labels, 'Brain stds', ymin=min_b_std, ymax= max_b_std)
        
    
        
    def make_radio_selector(self, parent, attrname, values, default):
        setattr(self, attrname, tk.StringVar(value=default))
        for v in values:
            tk.Radiobutton(parent, text=str(v), value=v,
                        variable=getattr(self, attrname)).pack(anchor="w")
            

    def on_close(self):
        print("Metrics window closed.")
        self.event_close.set()



