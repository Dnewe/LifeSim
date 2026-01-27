import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from queue import Empty
from utils.plot import *
import numpy as np
import pandas as pd

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
        self.root.title('Metrics')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # === Top info bar ===
        top = tk.Frame(self.root)
        top.pack(side='top', fill='x', padx=6, pady=6)
        self.step_label = tk.Label(top, text="STEP: 0", font=("Arial", 16))
        self.step_label.pack(side="left", padx=(0, 12))
        self.step_per_s_label = tk.Label(top, text="STEP PER S: 0", font=("Arial", 16))
        self.step_per_s_label.pack(side="left", padx=(0, 12))

        # === Main horizontal container ===
        main = tk.Frame(self.root)
        main.pack(side="top", fill="both", expand=True)
        
        # == Left column ==
        left = tk.Frame(main)
        left.pack(side="left", fill="y", expand=False, padx=4, pady=4)
        # General stats plots
        tk.Label(left, text="General Stats", font=("Arial", 14, "bold")).pack()
        self.left_fig1, self.left_axs1 = plt.subplots(1, 3, figsize=(6, 2))
        self.left_canvas1 = FigureCanvasTkAgg(self.left_fig1, master=left)
        self.left_canvas1.get_tk_widget().pack(fill="both", expand=False)
        # Species plot button
        tk.Label(left, text="Species", font=("Arial", 14, "bold")).pack()
        self.make_radio_selector(left, "species_plot_type", ['PCA scatter', 'density', 'dendrogram'], default= 'PCA scatter')
        # Species plot
        self.left_fig2 = plt.figure(figsize=(6, 6))
        self.left_ax2 = self.left_fig2.add_subplot(111)
        self.left_canvas2 = FigureCanvasTkAgg(self.left_fig2, master=left)
        self.left_canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        # == Middle column ==
        mid = tk.Frame(main)
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        # Genes distribution plots
        tk.Label(mid, text="Genes", font=("Arial", 14, "bold")).pack(pady=(10, 0))
        self.mid_fig1, self.mid_axs1 = plt.subplots(2, self.n_genes, figsize=(6, 3))
        self.mid_canvas1 = FigureCanvasTkAgg(self.mid_fig1, master=mid)
        self.mid_canvas1.get_tk_widget().pack(fill="both", expand=False)
        # Brain action distr
        tk.Label(mid, text="Brain", font=("Arial", 14, "bold")).pack(pady=(10, 0))
        self.mid_fig2 = plt.figure(figsize=(6, 2))
        self.mid_ax2 = self.mid_fig2.add_subplot(111)
        self.mid_canvas2 = FigureCanvasTkAgg(self.mid_fig2, master=mid)
        self.mid_canvas2.get_tk_widget().pack(fill="both", expand=False)
        # Brain distribution plots
        self.make_menu_selector(mid, "brain_action", self.actions, default= self.actions[0])
        self.mid_fig3, self.mid_axs3 = plt.subplots(1, 2, figsize=(6, 3))
        self.mid_canvas3 = FigureCanvasTkAgg(self.mid_fig3, master=mid)
        self.mid_canvas3.get_tk_widget().pack(fill="both", expand=False)
        
        # == Right column ==
        right = tk.Frame(main)
        right.pack(side="left", fill="y", expand=False, padx=4, pady=4)
        # button
        tk.Label(right, text="Specie details", font=("Arial", 14, "bold")).pack()
        self.right_button = tk.Button(right, text="(button)")
        self.right_button.pack(pady=6)
        # infos
        self.right_label = tk.Label(right, text="(some info here)")
        self.right_label.pack()
        # plots
        # species genesmeans
        self.right_fig1 = plt.figure(figsize=(5, 2))
        self.right_ax1 = self.right_fig1.add_subplot(111)
        self.right_canvas1 = FigureCanvasTkAgg(self.right_fig1, master=right)
        self.right_canvas1.get_tk_widget().pack(fill="both", expand=True)
        # species genes cvs
        self.right_fig2 = plt.figure(figsize=(5, 2))
        self.right_ax2 = self.right_fig2.add_subplot(111)
        self.right_canvas2 = FigureCanvasTkAgg(self.right_fig2, master=right)
        self.right_canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        self.update()
        self.root.mainloop()


    def update(self):
        try:
            metrics = self.queue.get_nowait()
        except Empty:
            metrics = None
        if metrics is not None:
            try:
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
        self.root.after(self.time_freq, self.update)
   
        
    def update_labels(self, metrics):
        self.step_label.config(text=f"STEP: {metrics.get("n_step", 0)}")
        self.step_per_s_label.config(text=f"FPS (cur|avg): {metrics.get("fps_cur", 0):.1f} | {metrics.get("fps_avg", 0):.1f}")
      
        
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
        species_plot_type = getattr(self, 'species_plot_type').get()
        if species_plot_type == 'PCA scatter' and metrics["species_scatter_data"] is not None:
            scatter_plot(self.left_ax2, **metrics["species_scatter_data"], title='Species scatterplot')
        elif species_plot_type == 'density' and metrics['species_density_data'] is not None:
            species_density_df = pd.DataFrame(metrics['species_density_data'])
            density_plot(self.left_ax2 , species_density_df, 'step', 'count', 'species', "Species proportions over time")
        elif species_plot_type == 'dendrogram':
            pass
        
        # == Middle ==
        # genes
        line_multiplot(self.mid_axs1[0], steps, metrics[f"genes_means"].values(), [f'{g} mean' for g in metrics[f"genes_means"].keys()])
        line_multiplot(self.mid_axs1[1], steps, metrics[f"genes_cvs"].values(), [f'{g} CV' for g in metrics[f"genes_cvs"].keys()])
        # brain 
        actions_density_df = pd.DataFrame(metrics['actions_density_data'])
        density_plot(self.mid_ax2, actions_density_df, 'step', 'count', 'action', "Actions density")
        brain_action = getattr(self, 'brain_action').get()
        action_means = metrics['brains_means'][brain_action]
        action_stds = metrics['brains_stds'][brain_action]
        multiline_plot(self.mid_axs3[0], steps, action_means.values(), action_means.keys(), f'{brain_action} mean')
        multiline_plot(self.mid_axs3[1], steps, action_stds.values(), action_stds.keys(), f'{brain_action} std')
        
        # == Right ==
        x = np.arange(self.n_genes)
        # species genes mean
        bar_plot(self.right_ax1, x, metrics["species_genes_mean"].values(), metrics["species_genes_mean"].keys(), self.genes, 'Genes means by species')
        # species genes cv
        bar_plot(self.right_ax2, x, metrics["species_genes_cv"].values(), metrics["species_genes_cv"].keys(), self.genes, 'Genes CVs by species')
        
    
        
    def make_radio_selector(self, parent, attrname, values, default):
        setattr(self, attrname, tk.StringVar(value=default))
        for v in values:
            tk.Radiobutton(parent, text=str(v), value=v,
                        variable=getattr(self, attrname)).pack(anchor="w")
            
    def make_menu_selector(self, parent, attrname, values, default):
        var = tk.StringVar(value=default)
        setattr(self, attrname, var)
        menubtn = tk.Menubutton(parent, textvariable=var, relief=tk.RAISED)
        menubtn.pack(anchor="w")
        menu = tk.Menu(menubtn, tearoff=0)
        for v in values:
            menu.add_radiobutton(
                label=str(v),
                value=v,
                variable=var
            )
        menubtn.config(menu=menu)

    def on_close(self):
        print("Metrics window closed.")
        self.event_close.set()



