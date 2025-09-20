# analisador_roleta_advanced.py
import streamlit as st
import numpy as np
from collections import Counter

# ---------------------------
# Configurações
# ---------------------------
FICHA_VALOR = 0.50  # valor de cada ficha
N_NUMEROS = 37
RODADA_ORDEM = [
    0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,
    16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26
]
VERMELHOS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
PRETOS = set(range(1,37)) - VERMELHOS

def cor_numero(n):
    if n == 0: return "zero"
    return "vermelho" if n in VERMELHOS else "preto"

def vizinhos_reais(numero, k):
    n = len(RODADA_ORDEM)
    idx = RODADA_ORDEM.index(numero)
    return [RODADA_ORDEM[(idx + i) % n] for i in range(-k, k+1)]

# ---------------------------
# Rede Neural NumPy
# ---------------------------
class FastMLP:
    def __init__(self, input_dim, hidden=64, lr=0.008, seed=1337):
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, np.sqrt(2/(input_dim+hidden)), size=(hidden, input_dim))
        self.b1 = np.zeros((hidden,1))
        self.W2 = rng.normal(0, np.sqrt(2/(hidden+N_NUMEROS)), size=(N_NUMEROS, hidden))
        self.b2 = np.zeros((N_NUMEROS,1))
        self.lr = lr

    def forward(self, x):
        z1 = self.W1.dot(x) + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2.dot(a1) + self.b2
        e = np.exp(z2 - np.max(z2, axis=0, keepdims=True))
        probs = e / np.sum(e, axis=0, keepdims=True)
        return {"a1":a1, "probs":probs}

    def predict_proba(self, x):
        x = x.reshape(-1,1) if x.ndim==1 else x
        return self.forward(x)['probs'][:,0]

    def train_step(self, x, y, epochs=1):
        x = x.reshape(-1,1)
        y_onehot = np.zeros((N_NUMEROS,1)); y_onehot[y,0]=1.0
        for _ in range(epochs):
            cache = self.forward(x)
            probs = cache['probs']
            dz2 = probs - y_onehot
            dW2 = dz2.dot(cache['a1'].T)
            db2 = dz2
            da1 = self.W2.T.dot(dz2)
            dz1 = da1 * (1 - np.tanh(cache['a1'])**2)
            dW1 = dz1.dot(x.T)
            db1 = dz1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

# ---------------------------
# Feature Builder
# ---------------------------
class FeatureBuilder:
    def __init__(self, lookback=8):
        self.lookback = lookback

    def build(self, history):
        L = self.lookback
        last = list(history[-L:])
        while len(last) < L:
            last.insert(0, -1)
        onehot = np.zeros((L*N_NUMEROS,), dtype=float)
        for i, num in enumerate(last):
            if 0 <= num <= 36: onehot[i*N_NUMEROS + num] = 1.0
        freq = np.zeros((N_NUMEROS,), dtype=float)
        if len(history) > 0:
            c = Counter(history); total=len(history)
            for k,v in c.items(): freq[k] = v/total
        counts = np.array([sum(1 for n in history if n in VERMELHOS),
                           sum(1 for n in history if n in PRETOS),
                           sum(1 for n in history if n==0),
                           sum(1 for n in history if n!=0 and n%2==0),
                           sum(1 for n in history if n%2==1)], dtype=float)
        if len(history) > 0: counts = counts/len(history)
        decay = np.zeros((N_NUMEROS,), dtype=float)
        for i, num in enumerate(reversed(history[-L:])):
            if 0 <= num <= 36: decay[num] += (i+1)/L
        if decay.sum()>0: decay=decay/decay.sum()
        feat = np.concatenate([onehot, freq, counts, decay])
        return feat

# ---------------------------
# Analisador Principal
# ---------------------------
class Analisador:
    def __init__(self):
        self.history = []
        self.fb = FeatureBuilder()
        input_dim = 8*N_NUMEROS + N_NUMEROS + 5 + N_NUMEROS
        self.model = FastMLP(input_dim=input_dim)
        self.logs = ""

    def inserir(self, numero, train=True, epochs=2):
        numero = int(numero)
        if len(self.history) >= 1 and train:
            x = self.fb.build(self.history)
            self.model.train_step(x, numero, epochs=epochs)
        self.history.append(numero)

    def probas(self):
        x = self.fb.build(self.history)
        return self.model.predict_proba(x)

    def recomendar(self, top_k=3, vizinhos_k=1):
        p = self.probas()
        idxs = np.argsort(p)[::-1][:top_k]
        recs=[]
        logs=[]
        for i in idxs:
            nums=vizinhos_reais(int(i), vizinhos_k)
            recs.append({"base":int(i), "prob":float(p[int(i)]), "conjunto":nums})
            logs.append(f"Número {i} (prob={p[int(i)]:.3f}) -> vizinhos: {nums} | custo: R${len(nums)*FICHA_VALOR:.2f}")
        self.logs="\n".join(logs)
        return recs

# ---------------------------
# Streamlit Interface
# ---------------------------
st.set_page_config(page_title="Analisador Roleta AI", layout="wide")
st.title("🌀 Analisador Roleta Europeia - AI Streamlit")

if "an" not in st.session_state: st.session_state.an = Analisador()
an = st.session_state.an

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Inserir número (0-36)")
    numero = st.number_input("Número da roleta", min_value=0, max_value=36, step=0)
    if st.button("Adicionar"):
        an.inserir(numero)
        st.experimental_rerun()

    viz_k = st.slider("Vizinhos (k)", min_value=1, max_value=9, value=1)
    top_k = st.slider("Top recomendações", min_value=1, max_value=5, value=3)

    if st.button("Limpar histórico"):
        an.history=[]
        st.experimental_rerun()

with col2:
    st.subheader("Histórico (últimos 50)")
    st.write(an.history[-50:][::-1])

    st.subheader("Recomendações AI")
    recs = an.recomendar(top_k=top_k, vizinhos_k=viz_k)
    for r in recs:
        st.markdown(f"- **Base {r['base']}** | Prob: {r['prob']:.3f} | Cobertura: {r['conjunto']} | Custo: R${len(r['conjunto'])*FICHA_VALOR:.2f}")

    st.subheader("Logs explicativos")
    st.text(an.logs)

st.subheader("Probabilidades completas")
probs = an.probas()
st.write({i:float(probs[i]) for i in range(N_NUMEROS)})
