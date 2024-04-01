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

        # Ensure delta is a 2D column vector with shape (1, hidden_dim)
        delta = delta.reshape(1, -1)
        self.x = self.x.reshape(1, -1)  # Shape (1, input_dim)
        self.hidden = self.hidden.reshape(1, -1)  # Shape (1, hidden_dim)

        # Initialize derivatives as zero arrays
        dx = np.zeros_like(self.x, dtype=np.float64)
        dh_prev_t = np.zeros_like(self.hidden, dtype=np.float64)

        # Compute the derivatives of the gates
        dz = delta * (self.hidden - self.n)  # # (1, hidden_dim) Derivative of loss w.r.t. update gate activation
        dr = delta * self.z * (1 - self.z) * (self.n * (1 - np.tanh(self.n)**2)) # Derivative w.r.t. reset gate
        dn = delta * (1 - self.z) * (1 - np.tanh(self.n)**2)  # Derivative w.r.t. candidate activation

        # Update dh_prev_t with the correctly shaped outputs
        dh_prev_t += dz.dot(self.Wzh.T) + dr.dot(self.Wrh.T) + (dn * self.r).dot(self.Wnh.T)

        print("dz shape:", dz.shape)
        print("Wzx shape:", self.Wzx.shape)
        print("dr shape:", dr.shape)
        print("Wrx shape:", self.Wrx.shape)
        print("dn shape:", dn.shape)
        print("Wnx shape:", self.Wnx.shape)

        # Compute the derivatives with respect to the input and weights
        dx = dz.dot(self.Wzx) + dr.dot(self.Wrx) + dn.dot(self.Wnx)
        
        self.dWrx = np.dot(dr.T, self.x)
        self.dWzx = np.dot(dz.T, self.x)
        self.dWnx = np.dot(dn.T, self.x)

        self.dWrh = np.dot(dr.T, self.hidden)
        self.dWzh = np.dot(dz.T, self.hidden)
        self.dWnh = np.dot((dn * self.r).T, self.hidden)

        # Calculate derivatives with respect to biases
        self.dbrx = np.sum(dr, axis=0)
        self.dbzx = np.sum(dz, axis=0)
        self.dbnx = np.sum(dn, axis=0)

        self.dbrh = self.dbrx
        self.dbzh = self.dbzx
        self.dbnh = np.sum(dn * self.r, axis=0)

        # Ensure the output shapes are correct
        dx = dx.reshape(-1)
        dh_prev_t = dh_prev_t.reshape(-1)

        return dx, dh_prev_t


        # delta = delta.reshape(-1, 1) #check col vector
        # delta = delta.reshape(self.h, -1)  # self.h is the size of the hidden dimension

        # # Ensure delta is a 2D column vector. If batch size is included, delta would have a shape (batch_size, hidden_size)
        # if delta.ndim == 1:
        #     delta = delta.reshape(-1, 1)  # Add a new axis to make it a 2D column vector.
        # elif delta.ndim == 2 and delta.shape[0] != self.h:
        #     delta = delta.T  # Transpose delta if the first dimension is not hidden_size.

        # # Ensure self.x and self.r have the correct shape. They should be 2D matrices with shapes (batch_size, input_size) and (batch_size, hidden_size), respectively.
        # if self.x.ndim == 1:
        #     self.x = self.x.reshape(1, -1)  # Reshape to 2D if it's currently 1D.
        # if self.r.ndim == 1:
        #     self.r = self.r.reshape(1, -1)  # Reshape to 2D if it's currently 1D.


        # self.dWrx =  np.dot(delta * self.r * (1 - self.r), self.x.T) #dL/dW_rx = dL/dr_t * dr_t/dW_rx
        # self.dWzx = np.dot(delta * self.z * (1 - self.z), self.x.T) #dL/dW_zx = dL/dz_t x dz_t/dW_zx
        # self.dWnx = np.dot(delta * (1 - np.tanh(self.n)**2), self.x.T) #dL/dW_nx = dL/dh_t x dh_t/dn_t
        # self.dWrh = np.dot(delta * self.r * (1 - self.r), self.h_prev_t.T) #dL/dW_rh = dL/dr_t x dr_t/dW_rh
        # self.dWzh = np.dot(delta * self.z * (1 - self.z), self.h_prev_t.T) #dL/dW_zh = dL/dz_t x dz_t/dW_zh
        # self.dWnh = np.dot(delta * (1 - np.tanh(self.n)**2), (self.r * self.h_prev_t).T) #dL/dW_nh = dL/dn_t x dz_t/dW_nh

        # self.dbrx = delta * self.r * (1 - self.r) #dL/db_rx = dL/dr_t x dz_t/db_rx
        # self.dbzx = delta * self.z * (1 - self.z) #dL/db_zx = dL/dz_t x dz_t/db_zx
        # self.dbnx = delta * (1 - np.tanh(self.n)**2) #dL/db_nx = dL/dn_t x dz_t/db_nx
        # self.dbrh = self.dbrx #dL/db_rh = dL/dr_t x dz_t/db_rh
        # self.dbzh = self.dbzx #dL/db_zh = dL/dz_t x dz_t/db_zh
        # self.dbnh = np.sum(delta * (1 - np.tanh(self.n)**2) * self.r, axis=0) #dL/db_nh = dL/dn_t x dz_t/db_nh

        # dx = (np.dot(delta * (1 - self.z), (1 - np.tanh(self.n)**2)), self.Wnx.T) + \
        #      (np.dot(delta * self.z * (1 - self.z), self.Wzx.T)) + \
        #      (np.dot(delta * self.r * (1 - self.r), self.Wrx.T))
                
        # dh_prev_t = np.dot(delta * (1 - self.z), (1 - np.tanh(self.n)**2) * self.r, self.Wnh.T) + \
        #     (np.dot(delta * self.z, np.eye(self.h))) + \
        #     (np.dot(delta * self.z * (1 - self.z), self.Wzh.T)) + \
        #     (np.dot(delta * self.r * (1 - self.r), self.Wrh.T))

        # assert dx.shape == (self.d,)
        # assert dh_prev_t.shape == (self.h,)

        # return dx, dh_prev_t
            
        # delta = delta.reshape(1, -1)
        # self.x = self.x.reshape(1, -1)  # Shape (1, input_dim)
        # self.hidden = self.hidden.reshape(1, -1)  # Shape (1, hidden_dim)
        # # Ensure all arrays involved in computations are of type float64
        # self.x = self.x.astype(np.float64)
        # self.hidden = self.hidden.astype(np.float64)
        # self.r = self.r.astype(np.float64)
        # self.z = self.z.astype(np.float64)
        # self.n = self.n.astype(np.float64)

        # # Compute the derivatives of the gates, ensuring the results are float64
        # dz = delta * (self.hidden - self.n).astype(np.float64)
        # dr = (delta * self.z * (1 - self.z) * self.n * (1 - np.tanh(self.n)**2) * self.Wnh).astype(np.float64)
        # dn = (delta * (1 - self.z) * (1 - np.tanh(self.n)**2)).astype(np.float64)

        #    # # Initialize derivatives as zero arrays
        # dx = np.zeros_like(self.x)
        # dh_prev_t = np.zeros_like(self.hidden)

        # # Update dh_prev_t with the correct data type
        # dh_prev_t += (dz * self.Wzh.T + dr * self.Wrh.T + dn * (self.r * self.Wnh).T).astype(np.float64)
        
        # # Compute the derivatives of the gates
        # dz = delta * (self.hidden - self.n)  # Derivative of loss w.r.t. update gate activation
        # dr = delta * self.z * (1 - self.z) * self.n * (1 - np.tanh(self.n)**2) * self.Wnh  # Derivative of loss w.r.t. reset gate activation
        # dn = delta * (1 - self.z) * (1 - np.tanh(self.n)**2)  # Derivative of loss w.r.t. candidate activation
        
        # Derivative of loss w.r.t. previous hidden state h_{t-1}
        # dh_prev_t += dz * self.Wzh.T + dr * self.Wrh.T + dn * (self.r * self.Wnh).T

        # Derivative of loss w.r.t. input x_t
        # dx += dz * self.Wzx.T + dr * self.Wrx.T + dn * self.Wnx.T

        # # Assuming dz, dr, and dn have shape (1, hidden_size) and self.r has shape (1, hidden_size

        # # Calculate derivatives with respect to weight matrices
        # self.dWrx = np.dot(dr.T, self.x)
        # self.dWzx = np.dot(dz.T, self.x)
        # self.dWnx = np.dot(dn.T, self.x)

        # self.dWrh = np.dot(dr.T, self.hidden)
        # self.dWzh = np.dot(dz.T, self.hidden)
        # self.dWnh = np.dot(dn.T, (self.r * self.hidden))

        # # Calculate derivatives with respect to biases
        # self.dbrx = np.sum(dr, axis=0)
        # self.dbzx = np.sum(dz, axis=0)
        # self.dbnx = np.sum(dn, axis=0)

        # # The biases for reset and update gates w.r.t. previous hidden state are the same as for the input
        # self.dbrh = self.dbrx
        # self.dbzh = self.dbzx
        # # Bias for candidate hidden state w.r.t. previous hidden state is different due to the element-wise multiplication by r
        # self.dbnh = np.sum(dn * self.r, axis=0)

        # # Reshape dx and dh_prev_t to remove the batch dimension if necessary
        # dx = dx.reshape(-1)
        # dh_prev_t = dh_prev_t.reshape(-1)

        # # Make sure the shapes of the calculated dWs and dbs match the initialized shapes of the respective Ws and bs
        # # ... ensure shape consistency here ...

        # return dx, dh_prev_t

            
        
        