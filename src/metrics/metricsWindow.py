import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from queue import Empty
import numpy as np

sns.set_theme()


CMAP = plt.get_cmap("Set3")

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

    def __init__(self, genes, queue, event_close, time_freq = 1000) -> None:
        self.genes = genes
        self.n_genes = len(genes)
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
        self.make_radio_selector(left, "species_plot_type", ['PCA scatter', 'dendrogram'], default= 'PCA scatter')
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
        self.mid_fig1, self.mid_axs1 = plt.subplots(2, self.n_genes, figsize=(6, 4))
        self.mid_canvas1 = FigureCanvasTkAgg(self.mid_fig1, master=mid)
        self.mid_canvas1.get_tk_widget().pack(fill="both", expand=False)
        # Brain distribution plots
        tk.Label(mid, text="Brain", font=("Arial", 14, "bold")).pack(pady=(10, 0))
        self.mid_fig2, self.mid_axs2 = plt.subplots(2, self.n_genes, figsize=(6, 4))
        self.mid_canvas2 = FigureCanvasTkAgg(self.mid_fig2, master=mid)
        self.mid_canvas2.get_tk_widget().pack(fill="both", expand=False)
        
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
            while True:
                metrics = self.queue.get_nowait()
                self.update_plots(metrics)
                self.update_labels(metrics)                  
        except Empty:
            pass
        
        self.left_canvas1.draw_idle()
        self.left_canvas2.draw_idle()
        self.mid_canvas1.draw_idle()
        self.mid_canvas2.draw_idle()
        self.right_canvas1.draw_idle()
        self.right_canvas2.draw_idle()
        self.root.after(self.time_freq, self.update)
        
    def clear_subaxs(self, subaxs):
        for ax in subaxs.ravel():
            ax.clear()
        
    def update_plots(self, metrics):
        # == Left ==
        
        # general stats
        self.clear_subaxs(self.left_axs1)
            # population
        self.left_axs1[0].set_title('Population')
        self.left_axs1[0].plot(metrics["population"])
        self.left_axs1[0].set_ylim(0)
            # births
        self.left_axs1[1].set_title('Births')
        self.left_axs1[1].plot(metrics["births"])
            # deaths
        self.left_axs1[2].set_title('Deaths')
        for reason in metrics["deaths"].keys():
            self.left_axs1[2].plot(metrics["deaths"][reason], label=reason)
        self.left_axs1[2].legend()
        
        # species plot
        self.left_ax2.clear()
        species_plot_type = getattr(self, 'species_plot_type').get()
        if species_plot_type == 'PCA scatter' and metrics["species_scatter_data_and_labels"] is not None:
            points, labels = metrics["species_scatter_data_and_labels"]
            colors = CMAP(labels % CMAP.N)
            self.left_ax2.set_title('Species scatterplot')
            self.left_ax2.scatter(points[:,0], points[:,1], color=colors)
        elif species_plot_type == 'dendrogram':
            '''z, cutoff = metrics["species_dendrogram_data_and_cutoff"]
            self.left_ax2.set_title('Species dendrogram')
            self.left_ax2.set_xlabel("Agents")
            self.left_ax2.set_ylabel("Distance")
            dendrogram(z, ax=self.left_ax2, color_threshold=cutoff, no_labels=True, count_sort=True)
            self.left_ax2.set_ylim(0, cutoff+0.5)
            self.left_ax2.axhline(cutoff, linestyle="--", color="gray", linewidth=1)'''
            

        
        # == Middle ==
        # genes
        self.clear_subaxs(self.mid_axs1)
        for i, gene in enumerate(self.genes):
            self.mid_axs1[0, i].set_title(f'{gene} mean')
            self.mid_axs1[1, i].set_title(f'{gene} CV')
            self.mid_axs1[0, i].plot(metrics[f"mean_{gene}"]) 
            self.mid_axs1[1, i].plot(metrics[f"cv_{gene}"]) 
        # brain 
        self.clear_subaxs(self.mid_axs2)
        for i, gene in []:
            self.mid_axs2[0, i].set_title(f'{gene} mean')
            self.mid_axs2[1, i].set_title(f'{gene} CV')
            self.mid_axs2[0, i].plot(metrics[f"mean_{gene}"]) 
            self.mid_axs2[1, i].plot(metrics[f"cv_{gene}"]) 
        
        # == Right ==
        # species genes mean
        self.right_ax1.clear()
        species_genes_mean = metrics["species_genes_mean"]
        x = np.arange(self.n_genes)
        width = 0.75 / (metrics['n_species'])
        multiplier = 0
        for s, means in species_genes_mean.items():
            color = CMAP(s % CMAP.N)
            offset = width * multiplier
            rects = self.right_ax1.bar(x + offset, means, width, label=s, color=color)
            self.right_ax1.bar_label(rects, padding=3)
            multiplier += 1
        self.right_ax1.set_title('Genes means by species')
        self.right_ax1.set_xticks(x + width, self.genes)
        self.right_ax1.set_ylabel('values')
        self.right_ax1.legend(loc='upper left', ncols=metrics['n_species'])
        # species genes cv
        self.right_ax2.clear()
        species_genes_cv = metrics["species_genes_cv"]
        x = np.arange(self.n_genes)
        width = 0.75 / (metrics['n_species'])
        multiplier = 0
        for s, cvs in species_genes_cv.items():
            color = CMAP(s % CMAP.N)
            offset = width * multiplier
            rects = self.right_ax2.bar(x + offset, cvs, width, label=s, color=color)
            self.right_ax2.bar_label(rects, padding=3)
            multiplier += 1
        self.right_ax2.set_title('Genes CVs by species')
        self.right_ax2.set_xticks(x + width, x)
        self.right_ax2.set_ylabel('values')
        self.right_ax2.legend(loc='upper left', ncols=metrics['n_species'])
        

    def update_labels(self, metrics):
        self.step_label.config(text=f"STEP: {metrics.get("n_step", 0)}")
        self.step_per_s_label.config(text=f"FPS (cur|avg): {metrics.get("fps_cur", 0):.1f} | {metrics.get("fps_avg", 0):.1f}")
        
    def make_radio_selector(self, parent, attrname, values, default):
        setattr(self, attrname, tk.StringVar(value=default))
        for v in values:
            tk.Radiobutton(parent, text=str(v), value=v,
                        variable=getattr(self, attrname)).pack(anchor="w")

    def on_close(self):
        print("Metrics window closed.")
        self.event_close.set()



