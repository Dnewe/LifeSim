import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from queue import Empty

sns.set_theme()


class MetricsWindow():

    def __init__(self, queue, event_close, time_freq = 1000) -> None:
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

        # === Container for plots ===
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side="top", fill="both", expand=True)
        
        # === Small subfigs (3x3) ===
        self.subfig, self.subaxs = plt.subplots(3, 4, figsize=(6, 6))
        self.sub_canvas = FigureCanvasTkAgg(self.subfig, master=plot_frame)
        self.sub_canvas.get_tk_widget().pack(side="left", fill="both", expand=False)

        # === Big scatter figure ===
        self.bigfig = plt.figure(figsize=(6, 6))
        self.bigax = self.bigfig.add_subplot(111)
        self.big_canvas = FigureCanvasTkAgg(self.bigfig, master=plot_frame)
        self.big_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

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
        
        self.sub_canvas.draw_idle()
        self.big_canvas.draw_idle()
        self.root.after(self.time_freq, self.update)

    def update_plots(self, metrics):
        # Subfigs
        for ax in self.subaxs.ravel():
                    ax.clear()
        self.subaxs[0, 0].plot(metrics["population"])
        self.subaxs[0, 1].plot(metrics["births"])
        self.subaxs[0, 2].plot(metrics["deaths"])
        self.subaxs[1, 0].plot(metrics["mean_morphology"])
        self.subaxs[1, 1].plot(metrics["mean_physiology"])
        self.subaxs[1, 2].plot(metrics["mean_sensorial"])
        self.subaxs[1, 3].plot(metrics["mean_hostilness"])
        self.subaxs[2, 0].plot(metrics["cv_morphology"])
        self.subaxs[2, 1].plot(metrics["cv_physiology"])
        self.subaxs[2, 2].plot(metrics["cv_sensorial"])
        self.subaxs[2, 3].plot(metrics["cv_hostilness"])
        # Bigfig
        points, labels = metrics["species_scatter_points_and_labels"]
        self.bigax.clear()
        self.bigax.scatter(points[:,0], points[:,1], c=labels, cmap="tab20")

    def update_labels(self, metrics):
        self.step_label.config(text=f"STEP: {metrics.get("n_step", 0)}")
        self.step_per_s_label.config(text=f"STEP_PER_SEC: {metrics.get("step_per_s", 0):.1f}")

    def on_close(self):
        print("Metrics window closed.")
        self.event_close.set()



