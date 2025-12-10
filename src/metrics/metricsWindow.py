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

        # label
        top = tk.Frame(self.root)
        top.pack(side='top', fill='x', padx=6, pady=6)

        self.step_label = tk.Label(top, text="STEP: 0", font=("Arial", 16))
        self.step_label.pack(side="left", padx=(0, 12))
        self.step_per_s_label = tk.Label(top, text="STEP PER S: 0", font=("Arial", 16))
        self.step_per_s_label.pack(side="left", padx=(0, 12))

        self.fig, self.axs = plt.subplots(3, 3, figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

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
        
        self.canvas.draw_idle()
        self.root.after(self.time_freq, self.update)

    def update_plots(self, metrics):
        for ax in self.axs.ravel():
                    ax.clear()
        self.axs[0, 0].plot(metrics["population"])
        self.axs[0, 1].plot(metrics["births"])
        self.axs[0, 2].plot(metrics["deaths"])
        self.axs[1, 0].plot(metrics["mean_morphology"])
        self.axs[1, 1].plot(metrics["mean_physiology"])
        self.axs[1, 2].plot(metrics["mean_sensorial"])
        self.axs[2, 0].plot(metrics["std_morphology"])
        self.axs[2, 1].plot(metrics["std_physiology"])
        self.axs[2, 2].plot(metrics["std_sensorial"])

    def update_labels(self, metrics):
        self.step_label.config(text=f"STEP: {metrics.get("n_step", 0)}")
        self.step_per_s_label.config(text=f"STEP_PER_SEC: {metrics.get("step_per_s", 0):.1f}")

    def on_close(self):
        print("Metrics window closed.")
        self.event_close.set()



