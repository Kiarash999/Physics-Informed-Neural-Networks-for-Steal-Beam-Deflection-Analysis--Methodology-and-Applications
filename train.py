from config import dde
from bcs import *
from ode import *
from analy import analy_cb, analy_frb, analy_frb_mspl



class CustomLossWeightCallback(dde.callbacks.Callback):
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        decay = np.exp(-0.01 * epoch)
        self.model.loss_weights[0] = 10 * decay  # ode
        self.model.loss_weights[1] = 10 * decay  # bcD_l
        self.model.loss_weights[2] = 10 * decay  # bcN_l
        self.model.loss_weights[3] = 10 * decay  # bcD_r
        self.model.loss_weights[4] = 10 * decay  # bcN_r



def solver (data, net, lr, iter, loss_weights=None, callback=None):
    
    
    model = dde.Model(data, net)

    model.compile('adam', lr, metrics=['l2 relative error'], loss_weights=loss_weights)
    history = model.train(iterations=iter, callbacks=callback)

    model.compile('L-BFGS',   metrics=['l2 relative error'], loss_weights=loss_weights)
    history = model.train(callbacks=callback)

    return model, history