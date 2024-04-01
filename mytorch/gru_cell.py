import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h) 
        self.bzx = np.random.randn(h) 
        self.bnx = np.random.randn(h) 

        self.brh = np.random.randn(h) 
        self.bzh = np.random.randn(h) 
        self.bnh = np.random.randn(h) 

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.n_state = np.dot(self.Wnh, h_prev_t) + self.bnh 
        
        self.r = self.r_act.forward(np.dot(self.Wrx, x) + self.brx + np.dot(self.Wrh, h_prev_t) + self.brh) 
        self.z = self.z_act.forward(np.dot(self.Wzx, x) + self.bzx + np.dot(self.Wzh, h_prev_t) + self.bzh) 
        self.n = self.h_act.forward(np.dot(self.Wnx, x) + self.bnx + self.r*(np.dot(self.Wnh, h_prev_t) + self.bnh)) 
        h_t = (1-self.z)*self.n + self.z*h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim) dL/dx_t
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim) dL/dh_{t-1}
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing. 


        # delta = delta.reshape(1, -1)  # (1, hidden_dim)
        self.x = self.x.reshape(1, -1)  
        self.hidden = self.hidden.reshape(1, -1) 
        self.r = self.r.reshape(1, -1)  
        self.z = self.z.reshape(1, -1)  
        self.n = self.n.reshape(1, -1)   

        dx = np.zeros_like(self.x, dtype=np.float64)
        dh_prev_t = np.zeros_like(self.hidden, dtype=np.float64)
        dn = np.zeros_like(self.n, dtype=np.float64)
        dz = np.zeros_like(self.z, dtype=np.float64)
        dr = np.zeros_like(self.r, dtype=np.float64)

        dz += delta * (self.hidden - self.n) 
        dn += delta * (1 - self.z)
        dh_prev_t += delta * self.z

        dn_actftn = dn * (1-self.n**2)
        dn_actftn_r = dn_actftn * self.r

        self.dWnx += np.dot(dn_actftn.T, self.x)
        dx += np.dot(dn_actftn, self.Wnx)
        self.dbnx += np.sum(dn_actftn, axis=0)
        dr += dn_actftn * self.n_state.T

        self.dWnh += np.dot(dn_actftn_r.T, self.hidden)
        dh_prev_t += np.dot(dn_actftn_r, self.Wnh)
        self.dbnh += np.sum(dn_actftn_r, axis=0)

        dz_actftn = dz * self.z * (1-self.z)

        dx += np.dot(dz_actftn, self.Wzx)
        self.dWzx += np.dot(dz_actftn.T, self.x)
        self.dWzh += np.dot(dz_actftn.T, self.hidden)
        dh_prev_t += np.dot(dz_actftn, self.Wzh)
        self.dbzx += np.sum(dz_actftn, axis=0)
        self.dbzh += np.sum(dz_actftn, axis=0)

        dr_actftn = dr * self.r * (1-self.r)
        dx += np.dot(dr_actftn, self.Wrx)
        self.dWrx += np.dot(dr_actftn.T, self.x)
        self.dWrh += np.dot(dr_actftn.T, self.hidden)
        dh_prev_t += np.dot(dr_actftn, self.Wrh)
        self.dbrx += np.sum(dr_actftn, axis=0)
        self.dbrh += np.sum(dr_actftn, axis=0)

        # dh_prev_t += dz.dot(self.Wzh.T) + dr.dot(self.Wrh.T) + dn.dot(self.Wnh.T)
        # dx = dz.dot(self.Wzx) + dr.dot(self.Wrx) + dn.dot(self.Wnx)

        # Ensure correct output shapes
        dx = dx.reshape(-1)
        dh_prev_t = dh_prev_t.reshape(-1)
        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t

