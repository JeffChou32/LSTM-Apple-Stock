import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging

RAW_URL = "https://raw.githubusercontent.com/AnishS04/CS4375_Term_Project/main/AAPL.csv"

### LOAD & PREP ###
df = pd.read_csv(RAW_URL, usecols=["Date", "Close"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

plt.plot(df.index, df["Close"])
plt.title("AAPL Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.tight_layout()
plt.show()

### WINDOWING ###
# makes sliding windows so we can say: [t-3, t-2, t-1] -> predict [t]
def df_to_windowed_df(dataframe: pd.DataFrame, first_date_str: str, last_date_str: str, n: int = 3) -> pd.DataFrame:
    first_ts = pd.to_datetime(first_date_str)                               # bounds as timestamps
    last_ts  = pd.to_datetime(last_date_str)
    idx = dataframe.index

    start_pos = max(idx.searchsorted(first_ts, side="left"), n)
    end_pos   = idx.searchsorted(last_ts, side="right")

    dates = []
    X, Y = [], []
    
    for pos in range(start_pos, min(end_pos, len(idx))):                    # iterate date index by integer position
        window = dataframe.iloc[pos - n : pos + 1]["Close"].to_numpy()      # [n history | target]
        x, y = window[:-1], window[-1]
        dates.append(idx[pos])
        X.append(x)
        Y.append(y)

    ret_df = pd.DataFrame({"Target Date": dates})
    X = np.array(X)
    for i in range(n):
        ret_df[f"Target-{n - i}"] = X[:, i]
    ret_df["Target"] = Y
    return ret_df

### BUILD WINDOWED DATASET ###
windowed_df = df_to_windowed_df(
    df,
    first_date_str="2018-03-25",        # start of the slice (inclusive)
    last_date_str="2019-03-25",         # end of the slice (exclusive)
    n=3                                 # use three previous closes to predict the next one
)


### NUMPY ARRAYS (FOR MODEL) ###
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)    # float32 faster

dates, X, y = windowed_df_to_date_X_y(windowed_df)

### SPLIT ###
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# visualize splits 
plt.figure()
plt.plot(dates_train, y_train)
plt.plot(dates_val,   y_val)
plt.plot(dates_test,  y_test)
plt.legend(['Train', 'Validation', 'Test'])
plt.xlabel('Date'); plt.ylabel('Close'); plt.title('AAPL splits')
plt.tight_layout()
plt.show()

### Implement LSTM ###
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
def dsigmoid(y): return y * (1.0 - y)            # y is sigmoid(x)
def dtanh(y): return 1.0 - y * y                 # y is tanh(x)
def relu(x): return np.maximum(0.0, x)
def drelu(x): return (x > 0).astype(x.dtype)

class LSTM:
    def __init__(self, input_dim=1, hidden=64, seed=42, clip=1.0):
        rng = np.random.default_rng(seed)
        self.I = input_dim
        self.H = hidden
        self.clip = clip

        Z = self.H + self.I
        scale = 1.0 / np.sqrt(Z)

        # LSTM gates
        self.Wf = rng.normal(0, scale, size=(self.H, Z)); self.bf = np.zeros(self.H)
        self.Wi = rng.normal(0, scale, size=(self.H, Z)); self.bi = np.zeros(self.H)
        self.Wo = rng.normal(0, scale, size=(self.H, Z)); self.bo = np.zeros(self.H)
        self.Wc = rng.normal(0, scale, size=(self.H, Z)); self.bc = np.zeros(self.H)

        # Dense head: H -> 32 -> 32 -> 1
        self.D1 = 32
        self.D2 = 32
        self.W1 = rng.normal(0, 1/np.sqrt(self.H), size=(self.D1, self.H)); self.b1 = np.zeros(self.D1)
        self.W2 = rng.normal(0, 1/np.sqrt(self.D1), size=(self.D2, self.D1)); self.b2 = np.zeros(self.D2)
        self.W3 = rng.normal(0, 1/np.sqrt(self.D2), size=(1, self.D2));      self.b3 = np.zeros(1)

    def _forward_sample(self, x):  # x: (T, I)
        T = x.shape[0]; H = self.H; I = self.I
        h = np.zeros((T, H)); c = np.zeros((T, H))  # hidden and cell states
        f = np.zeros((T, H)); i = np.zeros((T, H))  # save gate activations for backprop
        o = np.zeros((T, H)); g = np.zeros((T, H))  # concatenated [h_{t-1}, x_t]
        z = np.zeros((T, H + I))

        h_prev = np.zeros(H); c_prev = np.zeros(H)

        for t in range(T):
            z_t = np.concatenate([h_prev, x[t].reshape(-1)])
            z[t] = z_t
            f[t] = sigmoid(self.Wf @ z_t + self.bf)
            i[t] = sigmoid(self.Wi @ z_t + self.bi)
            o[t] = sigmoid(self.Wo @ z_t + self.bo)
            g[t] = np.tanh(self.Wc @ z_t + self.bc)

            c[t] = f[t] * c_prev + i[t] * g[t]
            h[t] = o[t] * np.tanh(c[t])

            h_prev = h[t]; c_prev = c[t]

        h_T = h[T-1]  # final hidden

        # MLP head adds a bit of nonlinearity capacity after the LSTM
        z1 = self.W1 @ h_T + self.b1        
        a1 = relu(z1)
        z2 = self.W2 @ a1 + self.b2       
        a2 = relu(z2)
        y_hat = (self.W3 @ a2 + self.b3).reshape(())            # scalar

        cache = (x, z, f, i, o, g, c, h, h_T, z1, a1, z2, a2)   # save for backward pass
        return y_hat, cache

    def _backward_sample(self, cache, dy):        
        x, z, f, i, o, g, c, h, h_T, z1, a1, z2, a2 = cache             # dy is dL/dy_hat — derivative of loss w.r.t. prediction for this sample
        T, H = h.shape; I = self.I
        
        dWf = np.zeros_like(self.Wf); dWi = np.zeros_like(self.Wi)      # grads init (same shapes as params)
        dWo = np.zeros_like(self.Wo); dWc = np.zeros_like(self.Wc)
        dbf = np.zeros_like(self.bf); dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo); dbc = np.zeros_like(self.bc)

        dW1 = np.zeros_like(self.W1); db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3); db3 = np.zeros_like(self.b3)

        # backprop through the dense head 
        g3 = float(dy)                                   
        dW3 += g3 * a2[None, :]                          
        db3 += np.array([g3])                            
        da2 = (self.W3.reshape(-1) * g3)                 
        dz2 = da2 * drelu(z2)                            

        dW2 += dz2[:, None] @ a1[None, :]                
        db2 += dz2                                       
        da1 = self.W2.T @ dz2                            

        dz1 = da1 * drelu(z1)                            
        dW1 += dz1[:, None] @ h_T[None, :]               
        db1 += dz1                                       
        dh_T = self.W1.T @ dz1                           

        # LSTM backward (through time)
        dh_next = dh_T
        dc_next = np.zeros(H)

        for t in reversed(range(T)):
            tanhc = np.tanh(c[t])
            do = dh_next * tanhc
            dc = dh_next * o[t] * dtanh(tanhc) + dc_next

            df = dc * (c[t-1] if t > 0 else 0.0)
            di = dc * g[t]
            dg = dc * i[t]
            dc_prev = dc * f[t]

            df_raw = df * dsigmoid(f[t])                    # gate local gradients
            di_raw = di * dsigmoid(i[t])
            do_raw = do * dsigmoid(o[t])
            dg_raw = dg * dtanh(g[t])

            z_t = z[t]

            dWf += np.outer(df_raw, z_t); dbf += df_raw     # accumulate param grads
            dWi += np.outer(di_raw, z_t); dbi += di_raw
            dWo += np.outer(do_raw, z_t); dbo += do_raw
            dWc += np.outer(dg_raw, z_t); dbc += dg_raw

            dz = (self.Wf.T @ df_raw                        # split gradient to h_{t-1} and x_t (we only need h_{t-1} here)
                + self.Wi.T @ di_raw
                + self.Wo.T @ do_raw
                + self.Wc.T @ dg_raw)

            dh_prev = dz[:H]                                # dx = dz[H:]  # not used           

            dh_next = dh_prev
            dc_next = dc_prev

        # clip all grads to avoid exploding updates
        for G in (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dW1, db1, dW2, db2, dW3, db3):
            np.clip(G, -self.clip, self.clip, out=G)

        return (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc,
                dW1, db1, dW2, db2, dW3, db3)

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, lr=1e-3, verbose=True):
        N, T, I = X.shape
        assert I == self.I                      # input dimension must match 

        for epoch in range(1, epochs+1):
            perm = np.random.permutation(N)     # randomize sample order each epoch
            total = 0.0
            for idx in perm:                    
                x_i = X[idx]; y_i = y[idx]

                y_hat, cache = self._forward_sample(x_i)
                err = y_hat - y_i
                total += 0.5 * (err ** 2)       # half MSE loss

                grads = self._backward_sample(cache, dy=err)

                (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc,
                 dW1, db1, dW2, db2, dW3, db3) = grads
                
                self.Wf -= lr * dWf; self.bf -= lr * dbf    # SGD update
                self.Wi -= lr * dWi; self.bi -= lr * dbi
                self.Wo -= lr * dWo; self.bo -= lr * dbo
                self.Wc -= lr * dWc; self.bc -= lr * dbc

                self.W1 -= lr * dW1; self.b1 -= lr * db1
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W3 -= lr * dW3; self.b3 -= lr * db3

            train_mse = total / N
            if verbose and (epoch % max(1, epochs//10) == 0 or epoch == 1):
                msg = f"Epoch {epoch}/{epochs} - train MSE: {train_mse:.6f}"
                if X_val is not None and y_val is not None:
                    val_preds = self.predict(X_val)
                    val_mse = np.mean((val_preds - y_val) ** 2)
                    msg += f" | val MSE: {val_mse:.6f}"
                print(msg)

    def predict(self, X):
        N = X.shape[0]
        out = np.zeros((N,))
        for n in range(N):
            y_hat, _ = self._forward_sample(X[n])
            out[n] = y_hat
        return out          # vector of predictions aligned with X

### TRAINING & EVALUATION ###
model = LSTM(input_dim=1, hidden=64, clip=1.0)
model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, verbose=True)

### PREDICTIONS & VISUALIZATION ###
train_predictions = model.predict(X_train)
val_predictions   = model.predict(X_val)
test_predictions  = model.predict(X_test)

plt.figure(); plt.plot(dates_train, train_predictions); plt.plot(dates_train, y_train)          # training curvve
plt.legend(['Training Predictions', 'Training Observations']); plt.tight_layout(); plt.show()

plt.figure(); plt.plot(dates_val, val_predictions); plt.plot(dates_val, y_val)                  # validation curve  
plt.legend(['Validation Predictions', 'Validation Observations']); plt.tight_layout(); plt.show()

plt.figure(); plt.plot(dates_test, test_predictions); plt.plot(dates_test, y_test)              # testing curve
plt.legend(['Testing Predictions', 'Testing Observations']); plt.tight_layout(); plt.show()

plt.figure()                                                                                    # combined plot
plt.plot(dates_train, train_predictions); plt.plot(dates_train, y_train)
plt.plot(dates_val,   val_predictions);   plt.plot(dates_val,   y_val)
plt.plot(dates_test,  test_predictions);  plt.plot(dates_test,  y_test)
plt.legend(['Training Predictions','Training Observations',
            'Validation Predictions','Validation Observations',
            'Testing Predictions','Testing Observations'])
plt.tight_layout(); plt.show()

### LOGGING ###
logging.basicConfig(filename='experiment_log.csv', level=logging.INFO, format='%(message)s')
logging.info("---------------\nParameters Chosen / Results")

def log_experiment(epochs, lr, X_train, y_train, X_test, y_test, train_predictions, test_predictions):      
    train_mse = mean_squared_error(y_train, train_predictions) 
    test_mse = mean_squared_error(y_test, test_predictions)

    train_rmse = np.sqrt(train_mse)  
    test_rmse = np.sqrt(test_mse)    
    
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    train_size = len(X_train)
    test_size = len(X_test)    
    
    train_test_split = f"{train_size}/{test_size}"      # Train/Test split ratio
    
    logging.info(f"epochs={epochs}, lr={lr}, "
                 f"train/test split={train_test_split}, dataset size={train_size + test_size}, "
                 f"training RMSE={train_rmse:.2f}, test RMSE={test_rmse:.2f}, "
                 f"training R2={train_r2:.2f}, test R2={test_r2:.2f}")

epochs = 100
lr = 1e-3

log_experiment(epochs, lr, X_train, y_train, X_test, y_test, train_predictions, test_predictions)