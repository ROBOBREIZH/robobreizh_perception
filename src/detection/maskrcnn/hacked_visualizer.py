from detectron2.utils.visualizer import Visualizer


class HackedVisualizer(Visualizer):
    def _jitter(self, color):
        return(color)