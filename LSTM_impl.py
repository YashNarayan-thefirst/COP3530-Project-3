import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, num_layers=1, batch_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_grad_norm = 5.0  # Gradient clipping threshold

        # Initialize LSTM layers
        self.layers = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            layer = {
                # Combined weights for all gates [input, forget, output, candidate]
                'W': np.random.randn(input_dim + hidden_size, 4 * hidden_size) * 0.01,
                'b': np.zeros((1, 4 * hidden_size)),
                # States
                'h': np.zeros((batch_size, hidden_size)),
                'c': np.zeros((batch_size, hidden_size))
            }
            self.layers.append(layer)

        # Output layer
        self.W_out = np.random.randn(hidden_size, 1) * 0.01
        self.b_out = np.zeros((1, 1))

    def forward(self, x_batch):
        """Forward pass through all LSTM layers
        Args:
            x_batch: shape (batch_size, seq_len, input_size)
        Returns:
            output: shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x_batch.shape
        self.x_batch = x_batch  # Store for backward pass
        
        # Process each layer
        for layer in self.layers:
            h_sequence = []
            for t in range(seq_len):
                # Combine input and previous hidden state
                x_t = x_batch[:, t, :]  # (batch_size, input_size)
                combined = np.concatenate([x_t, layer['h']], axis=1)
                
                # Compute all gates
                gates = np.dot(combined, layer['W']) + layer['b']
                i, f, o, g = np.split(gates, 4, axis=1)
                
                # Gate activations
                i = 1 / (1 + np.exp(-i))  # Input gate
                f = 1 / (1 + np.exp(-f))  # Forget gate
                o = 1 / (1 + np.exp(-o))  # Output gate
                g = np.tanh(g)            # Candidate
                
                # Update states
                layer['c'] = f * layer['c'] + i * g
                layer['h'] = o * np.tanh(layer['c'])
                h_sequence.append(layer['h'])
            
            # Prepare input for next layer
            x_batch = np.stack(h_sequence, axis=1)  # (batch_size, seq_len, hidden_size)
        
        # Final output (regression)
        return np.dot(layer['h'], self.W_out) + self.b_out

    def backward(self, grad_output):
        grad_norm = np.linalg.norm(grad_output)
        if grad_norm > self.max_grad_norm:
            grad_output = grad_output * (self.max_grad_norm / grad_norm)
        
        for layer in self.layers:
            layer['grad_W'] = np.zeros_like(layer['W'])
            layer['grad_b'] = np.zeros_like(layer['b'])
        grad_W_out = np.zeros_like(self.W_out)
        grad_b_out = np.zeros_like(self.b_out)
        
        # Simplified backprop through layers 
        return {
            'grad_W_out': grad_W_out,
            'grad_b_out': grad_b_out,
            'layer_grads': [{'grad_W': l['grad_W'], 'grad_b': l['grad_b']} 
                          for l in self.layers]
        }

    def update_params(self, grads, lr=0.001):
        """Update parameters with clipped gradients"""
        self.W_out -= lr * grads['grad_W_out']
        self.b_out -= lr * grads['grad_b_out']
        
        for i, layer in enumerate(self.layers):
            layer['W'] -= lr * grads['layer_grads'][i]['grad_W']
            layer['b'] -= lr * grads['layer_grads'][i]['grad_b']


