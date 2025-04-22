import numpy as np
"""
Custom LSTM class is made for demonstration purposes only. The keras implementation of LSTM significantly outperforms anything I could make. 
The results are reflected in the loss and training time.
"""
class LSTM:
    def __init__(self, input_size, hidden_size, num_layers=2, 
                 batch_size=32, checkpoint_interval=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.max_grad_norm = 5.0

        # Initialize layers
        self.layers = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            layer = {
                'W': np.random.randn(input_dim + hidden_size, 4 * hidden_size) * 0.01,
                'b': np.zeros((1, 4 * hidden_size)),
                'checkpoints': []
            }
            self.layers.append(layer)
        
        # Output layer
        self.W_out = np.random.randn(hidden_size, 1) * 0.01
        self.b_out = np.zeros((1, 1))

    def forward(self, x_batch):
        batch_size, seq_len, _ = x_batch.shape
        hiddens = []
        self.current_x_batch = x_batch
        
        for layer_idx, layer in enumerate(self.layers):
            layer_hiddens = []
            h = np.zeros((batch_size, self.hidden_size))
            c = np.zeros((batch_size, self.hidden_size))
            checkpoints = [(0, h.copy(), c.copy(), x_batch.copy())]
            
            for t in range(seq_len):
                if t % self.checkpoint_interval == 0 and t != 0:
                    checkpoints.append((t, h.copy(), c.copy(), x_batch.copy()))
                
                input_dim = self.input_size if layer_idx == 0 else self.hidden_size
                combined = np.concatenate([x_batch[:,t,:], h], axis=1)
                gates = np.dot(combined, layer['W']) + layer['b']
                i, f, g, o = np.split(gates, 4, axis=1)
                
                i = 1 / (1 + np.exp(-i))
                f = 1 / (1 + np.exp(-f))
                o = 1 / (1 + np.exp(-o))
                g = np.tanh(g)
                
                c = f * c + i * g
                h = o * np.tanh(c)
                layer_hiddens.append(h)
            
            layer['checkpoints'] = checkpoints
            x_batch = np.stack(layer_hiddens, axis=1)
            hiddens.append(h)
        
        return np.dot(hiddens[-1], self.W_out) + self.b_out
    
    def backward(self, dout):
        grads = {
            'W_out': np.zeros_like(self.W_out),
            'b_out': np.zeros_like(self.b_out),
            'layers': [{'W': np.zeros_like(layer['W']), 'b': np.zeros_like(layer['b'])} 
                    for layer in self.layers]
        }
        
        batch_size = dout.shape[0]
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Output layer gradients
        grads['W_out'] = np.dot(self.layers[-1]['checkpoints'][-1][1].T, dout)
        grads['b_out'] = np.sum(dout, axis=0, keepdims=True)
        dh_next = np.dot(dout, self.W_out.T)
        
        for l_idx in reversed(range(self.num_layers)):
            layer = self.layers[l_idx]
            checkpoints = layer['checkpoints']
            input_dim = self.input_size if l_idx == 0 else self.hidden_size  # NEW: Dynamic input_dim
            
            dW = np.zeros_like(layer['W'])
            db = np.zeros_like(layer['b'])
            
            for c_idx in reversed(range(len(checkpoints)-1)):
                start_t, h_start, c_start, x_segment = checkpoints[c_idx]
                end_t = checkpoints[c_idx+1][0]
                
                h = h_start.copy()
                c = c_start.copy()
                cache = []

                # Forward pass to rebuild cache
                for t in range(start_t, end_t):
                    combined = np.concatenate([x_segment[:,t,:], h], axis=1)
                    gates = np.dot(combined, layer['W']) + layer['b']
                    i, f, o, g = np.split(gates, 4, axis=1)
                    
                    i = 1 / (1 + np.exp(-i))
                    f = 1 / (1 + np.exp(-f))
                    o = 1 / (1 + np.exp(-o))
                    g = np.tanh(g)
                    
                    c = f * c + i * g
                    h = o * np.tanh(c)
                    cache.append((combined, i, f, o, g, c.copy()))
                
                # Backward pass
                dh = dh_next.copy()
                dc = dc_next.copy()
                
                for t in reversed(range(len(cache))):
                    combined, i, f, o, g, c_prev = cache[t]
                    
                    # Gradients of h and c
                    do = dh * np.tanh(c_prev)
                    do_raw = o * (1 - o) * do
                    
                    dc = dh * o * (1 - np.tanh(c_prev)**2) + dc
                    di = dc * g
                    di_raw = i * (1 - i) * di
                    
                    dg = dc * i
                    dg_raw = (1 - g**2) * dg
                    
                    df = dc * c_prev
                    df_raw = f * (1 - f) * df
                    
                    dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)
                    
                    # Update gradients
                    dW += np.dot(combined.T, dgates)
                    db += np.sum(dgates, axis=0, keepdims=True)
                    
                    # Split dxh correctly using input_dim
                    dxh = np.dot(dgates, layer['W'].T)
                    dx = dxh[:, :input_dim]  # CORRECTED: Use input_dim
                    dh = dxh[:, input_dim:input_dim + self.hidden_size]  # CORRECTED
                    
                    dc = dc * f
                
                # Average gradients over the segment
                grads['layers'][l_idx]['W'] = dW / (end_t - start_t)
                grads['layers'][l_idx]['b'] = db / (end_t - start_t)
                
                dh_next = dh
                dc_next = dc
        
        return grads
    
    def train_step(self, x_batch, y_batch, lr=0.001):
        y_pred = self.forward(x_batch)
        loss = float(np.mean((y_pred - y_batch)**2))
        dout = 2 * (y_pred - y_batch) / self.batch_size
        
        grads = self.backward(dout)
        self._apply_gradients(grads, lr)
        return loss

    def eval_step(self, x_batch, y_batch):
        y_pred = self.forward(x_batch)
        return float(np.mean((y_pred - y_batch)**2))

    def _apply_gradients(self, grads, lr):
        # Gradient clipping
        total_norm = 0
        for layer_grad in grads['layers']:
            total_norm += np.sum(layer_grad['W']**2) + np.sum(layer_grad['b']**2)
        total_norm += np.sum(grads['W_out']**2) + np.sum(grads['b_out']**2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            for layer_grad in grads['layers']:
                layer_grad['W'] *= scale
                layer_grad['b'] *= scale
            grads['W_out'] *= scale
            grads['b_out'] *= scale
        
        # Update weights
        self.W_out -= lr * grads['W_out']
        self.b_out -= lr * grads['b_out']
        for l_idx, layer in enumerate(self.layers):
            layer['W'] -= lr * grads['layers'][l_idx]['W']
            layer['b'] -= lr * grads['layers'][l_idx]['b']

    def save(self, filepath):
        layer_params = {}
        for i, layer in enumerate(self.layers):
            layer_params[f'W_layer{i}'] = layer['W']
            layer_params[f'b_layer{i}'] = layer['b']
        np.savez(filepath, **layer_params, W_out=self.W_out, b_out=self.b_out)

    def load(self, filepath):
        data = np.load(filepath)
        self.W_out = data['W_out']
        self.b_out = data['b_out']
        for i in range(self.num_layers):
            self.layers[i]['W'] = data[f'W_layer{i}']
            self.layers[i]['b'] = data[f'b_layer{i}']

