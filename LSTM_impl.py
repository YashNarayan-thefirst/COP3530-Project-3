import numpy as np
class LSTM:
    def __init__(self, input_size, hidden_size, num_layers=2,
             batch_size=32, checkpoint_interval=5, dropout=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.max_grad_norm = 5.0
        self.dropout = dropout

        self.epsilon = 1e-7 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

        self.layers = []
        self.m = []
        self.v = []

        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size

            # Glorot uniform for input part
            glorot_limit = np.sqrt(6.0 / (input_dim + hidden_size))
            W_input = np.random.uniform(-glorot_limit, glorot_limit, (input_dim, 4 * hidden_size))

            # Orthogonal for recurrent part
            def orthogonal(shape):
                a = np.random.randn(*shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                return u if u.shape == shape else v

            W_recurrent = orthogonal((hidden_size, 4 * hidden_size))

            # Concatenate input and recurrent weights
            W = np.concatenate([W_input, W_recurrent], axis=0)

            # Bias: zeros except forget gate = 1
            b = np.zeros((1, 4 * hidden_size))
            b[0, hidden_size:2*hidden_size] = 1.0

            self.layers.append({'W': W, 'b': b, 'checkpoints': []})
            self.m.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})
            self.v.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})

        # Output layer: Glorot uniform
        limit_out = np.sqrt(6 / (hidden_size + 1))
        self.W_out = np.random.uniform(-limit_out, limit_out, (hidden_size, 1))
        self.b_out = np.zeros((1, 1))

        self.m_out = {'W': np.zeros_like(self.W_out), 'b': np.zeros_like(self.b_out)}
        self.v_out = {'W': np.zeros_like(self.W_out), 'b': np.zeros_like(self.b_out)}

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
                o = np.clip(o, -100, 100) 
                f = np.clip(f, -100, 100)
                i = np.clip(i, -100, 100)
                g = np.clip(g, -100, 100)
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
                    i = np.clip(i, -100, 100)
                    f = np.clip(f, -100, 100)
                    o = np.clip(o, -100, 100)
                    g = np.clip(g, -100, 100)
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
        dout = 2 * (y_pred - y_batch) / y_batch.shape[0]
        
        grads = self.backward(dout)
        self._apply_adam(grads, lr)
        return loss

    def _layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _apply_dropout(self, x):
        if self.dropout > 0.0:
            mask = (np.random.rand(*x.shape) >= self.dropout).astype(np.float32)
            return (x * mask) / (1.0 - self.dropout)
        return x

    def _apply_mask(self, x, mask):
        return x * mask[:, :, np.newaxis]

    def eval_step(self, x_batch, y_batch):
        y_pred = self.forward(x_batch)
        return float(np.mean((y_pred - y_batch)**2))

    def _apply_adam(self, grads, lr):
        self.t += 1
        for l_idx, layer in enumerate(self.layers):
            for param in ['W', 'b']:
                self.m[l_idx][param] = self.beta1 * self.m[l_idx][param] + (1 - self.beta1) * grads['layers'][l_idx][param]
                self.v[l_idx][param] = self.beta2 * self.v[l_idx][param] + (1 - self.beta2) * (grads['layers'][l_idx][param] ** 2)

                m_hat = self.m[l_idx][param] / (1 - self.beta1 ** self.t)
                v_hat = self.v[l_idx][param] / (1 - self.beta2 ** self.t)

                layer[param] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        for param in ['W', 'b']:
            self.m_out[param] = self.beta1 * self.m_out[param] + (1 - self.beta1) * grads[f'{param}_out']
            self.v_out[param] = self.beta2 * self.v_out[param] + (1 - self.beta2) * (grads[f'{param}_out'] ** 2)

            m_hat = self.m_out[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v_out[param] / (1 - self.beta2 ** self.t)

            if param == 'W':
                self.W_out -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                self.b_out -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def save(self, filename):
        save_dict = {
        'W_out': self.W_out,
        'b_out': self.b_out,
        'num_layers': self.num_layers,
        'input_size': self.input_size,
        'hidden_size': self.hidden_size
        }
        for i, layer in enumerate(self.layers):
            save_dict[f'W_layer{i}'] = layer['W']
            save_dict[f'b_layer{i}'] = layer['b']
        save_dict['t'] = self.t
        for i in range(self.num_layers):
            save_dict[f'm_W_layer{i}'] = self.m[i]['W']
            save_dict[f'm_b_layer{i}'] = self.m[i]['b']
            save_dict[f'v_W_layer{i}'] = self.v[i]['W']
            save_dict[f'v_b_layer{i}'] = self.v[i]['b']
        
        save_dict['m_W_out'] = self.m_out['W']
        save_dict['m_b_out'] = self.m_out['b']
        save_dict['v_W_out'] = self.v_out['W']
        save_dict['v_b_out'] = self.v_out['b']
        
        np.savez(filename, **save_dict)

    def load(self, filepath):
        data = np.load(filepath)
        self.W_out = data['W_out']
        self.b_out = data['b_out']
        for i in range(self.num_layers):
            self.layers[i]['W'] = data[f'W_layer{i}']
            self.layers[i]['b'] = data[f'b_layer{i}']

    def fit(self, X_train, y_train, epochs=10, batch_size=32, lr=0.001, verbose=True):
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                loss = self.train_step(x_batch, y_batch, lr)
                epoch_loss += loss * x_batch.shape[0]  
            epoch_loss /= n_samples
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        return self.eval_step(X, y)

    def eval_step(self, x_batch, y_batch):
        y_pred = self.forward(x_batch)
        return np.mean((y_pred - y_batch)**2)
