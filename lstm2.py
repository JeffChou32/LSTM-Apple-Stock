import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_URL = "https://raw.githubusercontent.com/AnishS04/CS4375_Term_Project/main/AAPL.csv"

# --- Load & prep ---
df = pd.read_csv(RAW_URL, usecols=["Date", "Close"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

print("Data loaded:", len(df), "rows")
print(df.head().to_string())

# --- Plot ---
plt.plot(df.index, df["Close"])
plt.title("AAPL Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.tight_layout()
plt.show()

# --- Windowing util (robust, index-position based) ---
def df_to_windowed_df(dataframe: pd.DataFrame, first_date_str: str, last_date_str: str, n: int = 3) -> pd.DataFrame:
    """
    Builds a supervised learning dataset with n previous 'Close' values predicting the next 'Close'.
    Walks the DatetimeIndex by position; no empty slices; aligns to trading days.
    """
    first_ts = pd.to_datetime(first_date_str)
    last_ts  = pd.to_datetime(last_date_str)

    idx = dataframe.index

    # Start at the first row >= first_ts, but ensure we have n prior rows for the window
    start_pos = max(idx.searchsorted(first_ts, side="left"), n)
    # Stop at the first row strictly after last_ts
    end_pos   = idx.searchsorted(last_ts, side="right")

    dates = []
    X, Y = [], []

    for pos in range(start_pos, min(end_pos, len(idx))):
        window = dataframe.iloc[pos - n : pos + 1]["Close"].to_numpy()
        if window.shape[0] != n + 1:
            continue  # safety guard
        x, y = window[:-1], window[-1]
        dates.append(idx[pos])
        X.append(x)
        Y.append(y)

    if not dates:
        print("No rows produced. Check your date range or reduce n.")
        return pd.DataFrame(columns=["Target Date"] + [f"Target-{i}" for i in range(n, 0, -1)] + ["Target"])

    ret_df = pd.DataFrame({"Target Date": dates})
    X = np.array(X)
    for i in range(n):
        ret_df[f"Target-{n - i}"] = X[:, i]
    ret_df["Target"] = Y
    return ret_df

# --- Build windowed dataset ---
windowed_df = df_to_windowed_df(
    df,
    first_date_str="2018-03-25",
    last_date_str="2019-03-25",
    n=3
)

print("\nWindowed dataset preview:")
print(windowed_df.head().to_string(index=False))
print("\nRows in windowed_df:", len(windowed_df))

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

dates.shape, X.shape, y.shape

q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

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

class NumpyLSTM_Dense:
    """
    LSTM (hidden=64) -> Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(1)
    Input shape per sample: (T=3, I=1)
    """
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
        h = np.zeros((T, H)); c = np.zeros((T, H))
        f = np.zeros((T, H)); i = np.zeros((T, H))
        o = np.zeros((T, H)); g = np.zeros((T, H))
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

        # Dense head
        z1 = self.W1 @ h_T + self.b1           # (32,)
        a1 = relu(z1)
        z2 = self.W2 @ a1 + self.b2            # (32,)
        a2 = relu(z2)
        y_hat = (self.W3 @ a2 + self.b3).reshape(())  # scalar

        cache = (x, z, f, i, o, g, c, h, h_T, z1, a1, z2, a2)
        return y_hat, cache

    def _backward_sample(self, cache, dy):
        # dy = dL/dy_hat (scalar)
        x, z, f, i, o, g, c, h, h_T, z1, a1, z2, a2 = cache
        T, H = h.shape; I = self.I

        # grads init
        dWf = np.zeros_like(self.Wf); dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo); dWc = np.zeros_like(self.Wc)
        dbf = np.zeros_like(self.bf); dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo); dbc = np.zeros_like(self.bc)

        dW1 = np.zeros_like(self.W1); db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3); db3 = np.zeros_like(self.b3)

        # Dense head backward
        g3 = float(dy)                                   # ensure scalar float
        dW3 += g3 * a2[None, :]                          # (1,32)
        db3 += np.array([g3])                            # (1,)
        da2 = (self.W3.reshape(-1) * g3)                 # (32,)
        dz2 = da2 * drelu(z2)                            # (32,)

        dW2 += dz2[:, None] @ a1[None, :]                # (32,32)
        db2 += dz2                                       # (32,)
        da1 = self.W2.T @ dz2                            # (32,)

        dz1 = da1 * drelu(z1)                            # (32,)
        dW1 += dz1[:, None] @ h_T[None, :]               # (32,H)
        db1 += dz1                                       # (32,)
        dh_T = self.W1.T @ dz1                           # (H,)

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

            df_raw = df * dsigmoid(f[t])
            di_raw = di * dsigmoid(i[t])
            do_raw = do * dsigmoid(o[t])
            dg_raw = dg * dtanh(g[t])

            z_t = z[t]

            dWf += np.outer(df_raw, z_t); dbf += df_raw
            dWi += np.outer(di_raw, z_t); dbi += di_raw
            dWo += np.outer(do_raw, z_t); dbo += do_raw
            dWc += np.outer(dg_raw, z_t); dbc += dg_raw

            dz = (self.Wf.T @ df_raw
                + self.Wi.T @ di_raw
                + self.Wo.T @ do_raw
                + self.Wc.T @ dg_raw)

            dh_prev = dz[:H]
            # dx = dz[H:]  # not used

            dh_next = dh_prev
            dc_next = dc_prev

        # clip
        for G in (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dW1, db1, dW2, db2, dW3, db3):
            np.clip(G, -self.clip, self.clip, out=G)

        return (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc,
                dW1, db1, dW2, db2, dW3, db3)

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, lr=1e-3, verbose=True):
        N, T, I = X.shape
        assert I == self.I

        for epoch in range(1, epochs+1):
            perm = np.random.permutation(N)
            total = 0.0
            for idx in perm:
                x_i = X[idx]; y_i = y[idx]

                y_hat, cache = self._forward_sample(x_i)
                err = y_hat - y_i
                total += 0.5 * (err ** 2)

                grads = self._backward_sample(cache, dy=err)

                (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc,
                 dW1, db1, dW2, db2, dW3, db3) = grads

                # SGD update
                self.Wf -= lr * dWf; self.bf -= lr * dbf
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
        return out

# ===== Train =====
model = NumpyLSTM_Dense(input_dim=1, hidden=64, clip=1.0)
model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, verbose=True)

# ===== Predictions & plots =====
train_predictions = model.predict(X_train)
val_predictions   = model.predict(X_val)
test_predictions  = model.predict(X_test)

plt.figure(); plt.plot(dates_train, train_predictions); plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations']); plt.tight_layout(); plt.show()

plt.figure(); plt.plot(dates_val, val_predictions); plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations']); plt.tight_layout(); plt.show()

plt.figure(); plt.plot(dates_test, test_predictions); plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations']); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(dates_train, train_predictions); plt.plot(dates_train, y_train)
plt.plot(dates_val,   val_predictions);   plt.plot(dates_val,   y_val)
plt.plot(dates_test,  test_predictions);  plt.plot(dates_test,  y_test)
plt.legend(['Training Predictions','Training Observations',
            'Validation Predictions','Validation Observations',
            'Testing Predictions','Testing Observations'])
plt.tight_layout(); plt.show()


from sklearn.metrics import mean_squared_error, r2_score
import logging

# --- Initialize logging ---
logging.basicConfig(filename='experiment_log.csv', level=logging.INFO)

# --- Log Header (one-time write) ---
logging.info("Experiment Log\n---------------\nExperiment Number, Parameters Chosen, Results")

# --- Training & Evaluation --- (for each experiment)
def log_experiment(experiment_number, epochs, lr, batch_size, X_train, y_train, X_test, y_test, train_predictions, test_predictions):
    # Calculate RMSE for training and test sets
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    train_rmse = np.sqrt(train_mse)  # Manual calculation of RMSE
    test_rmse = np.sqrt(test_mse)    # Manual calculation of RMSE
    
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Dataset sizes (can be adjusted to match your actual data)
    train_size = len(X_train)
    test_size = len(X_test)
    
    # Train/Test split ratio
    train_test_split = f"{train_size}/{test_size}"

    # Log experiment details
    logging.info(f"{experiment_number}, epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                 f"train/test split={train_test_split}, dataset size={train_size + test_size}, "
                 f"training RMSE={train_rmse:.2f}, test RMSE={test_rmse:.2f}, "
                 f"training R²={train_r2:.2f}, test R²={test_r2:.2f}")

# Example for logging after an experiment:
experiment_number = 1
epochs = 100
lr = 1e-3
batch_size = 64

# Assuming model training and prediction has already been performed:
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

log_experiment(experiment_number, epochs, lr, batch_size, X_train, y_train, X_test, y_test, train_predictions, test_predictions)