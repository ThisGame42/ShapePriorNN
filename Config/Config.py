import os


class Config(object):
    def __init__(self,
                 lr,
                 w_decay,
                 device,
                 num_classes,
                 prob):
        self.lr = lr
        self.w_decay = w_decay
        self.num_classes = num_classes
        self.prob = prob
        self.device = device
        self._set_paths()


    def _set_paths(self):
        if os.name.startswith("nt"):
            # Windows
            self.path_model = "F:\\FIX_ME_LATER"
            self.path_plot = "F:\\FIX_ME_LATER"
        else:
            # Linux
            self.path_model = "/FIX/ME/LATER"
            self.path_plot = "/FIX/ME/LATER"