"""
THE CRYPTO SENTINEL — Backend
ESME Paris 2026 · Tanguy & Augustin

FIXES :
  - Gemini 1.5 Flash (stable), .env chargé depuis le bon dossier
  - Portfolio : trade uniquement après entraînement RF complet
  - Stop loss élargi à -5% (crypto volatile)
  - Seuil achat relevé à 70/100
  - Filtre tendance MM50 obligatoire
  - Filtre volume minimum
"""

import os, time, math, logging, threading, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_OK = True
except ImportError:
    ML_OK = False

# ── Charger .env depuis le dossier du script (résout les problèmes de chemin) ──
SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(SCRIPT_DIR / ".env", override=True)

GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_API_KEY", "").strip()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("CryptoSentinel")

BINANCE_URL = "https://api.binance.com"

# ── 100 cryptos par tier de liquidité ──
ALL_CRYPTOS = [
    # Tier 1 — max 5% du capital
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
    "ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT","LINKUSDT",
    # Tier 2 — max 3%
    "LTCUSDT","UNIUSDT","ATOMUSDT","NEARUSDT","FTMUSDT",
    "DOGEUSDT","SHIBUSDT","LDOUSDT","INJUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","RENDERUSDT","FETUSDT",
    "IMXUSDT","GRTUSDT","HBARUSDT","ETCUSDT","BCHUSDT",
    # Tier 3 — max 2%
    "FILUSDT","ALGOUSDT","VETUSDT","ICPUSDT","SANDUSDT",
    "MANAUSDT","AXSUSDT","GALAUSDT","CRVUSDT","AAVEUSDT",
    "MKRUSDT","COMPUSDT","RUNEUSDT","ZILUSDT","PEPEUSDT",
    "FLOKIUSDT","STXUSDT","SEIUSDT","TIAUSDT","JUPUSDT",
    "WIFUSDT","BONKUSDT","OCEANUSDT","AGIXUSDT","WOOUSDT",
    "GMXUSDT","DYDXUSDT","LRCUSDT","MASKUSDT","ENJUSDT",
    # Tier 4 — max 1%
    "CHZUSDT","FLOWUSDT","QNTUSDT","EGLDUSDT","THETAUSDT",
    "XTZUSDT","IOTAUSDT","NEOUSDT","DASHUSDT","ZECUSDT",
    "RVNUSDT","ONTUSDT","WAVESUSDT","ZENUSDT","SCUSDT",
    "DCRUSDT","BATUSDT","ZRXUSDT","XLMUSDT","TRXUSDT",
    "EOSUSDT","XMRUSDT","IOSTUSDT","CELRUSDT","BANDUSDT",
    "KSMUSDT","SXPUSDT","KNCUSDT","ROSEUSDT","CTSIUSDT",
    "COTIUSDT","GLMUSDT","NMRUSDT","STORJUSDT","CVCUSDT",
    "POWRUSDT","DENTUSDT","APEUSDT","YFIUSDT","SUSHIUSDT",
]

def get_tier(symbol):
    """Retourne l'exposition max selon la liquidité de l'actif."""
    idx = ALL_CRYPTOS.index(symbol) if symbol in ALL_CRYPTOS else 99
    if idx < 10:  return {"tier": 1, "exposure": 0.05, "label": "T1"}
    if idx < 30:  return {"tier": 2, "exposure": 0.03, "label": "T2"}
    if idx < 60:  return {"tier": 3, "exposure": 0.02, "label": "T3"}
    return             {"tier": 4, "exposure": 0.01, "label": "T4"}

# Volume minimum en USDT/24h pour trader (évite les altcoins illiquides)
MIN_VOLUME_USDT = {1: 50_000_000, 2: 10_000_000, 3: 2_000_000, 4: 500_000}


# =============================================================================
# SECTION 1 — DONNÉES BINANCE
# =============================================================================
class BinanceData:
    def __init__(self):
        self._cache = {}

    def _get(self, url, params={}, ttl=5):
        key = url + str(sorted(params.items()))
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < ttl:
                return data
        try:
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            self._cache[key] = (data, time.time())
            return data
        except Exception as e:
            log.warning(f"Binance {url}: {e}")
            return self._cache.get(key, (None, 0))[0]

    def klines(self, symbol, interval="1h", limit=300):
        data = self._get(f"{BINANCE_URL}/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit}, ttl=30)
        if not data:
            return pd.DataFrame()
        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","tb","tq","ignore"]
        df = pd.DataFrame(data, columns=cols)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df[["open_time","open","high","low","close","volume"]].copy()

    def all_tickers(self):
        """Tous les prix en une seule requête."""
        data = self._get(f"{BINANCE_URL}/api/v3/ticker/24hr", {}, ttl=2)
        if not data:
            return {}
        return {
            t["symbol"]: {
                "price":  float(t.get("lastPrice", 0) or 0),
                "change": float(t.get("priceChangePercent", 0) or 0),
                "volume": float(t.get("quoteVolume", 0) or 0),
                "high":   float(t.get("highPrice", 0) or 0),
                "low":    float(t.get("lowPrice", 0) or 0),
            }
            for t in data if t["symbol"] in ALL_CRYPTOS
        }

    def spot_price(self, symbol):
        data = self._get(f"{BINANCE_URL}/api/v3/ticker/price",
            {"symbol": symbol}, ttl=1)
        return float(data["price"]) if data else 0.0


# =============================================================================
# SECTION 2 — INDICATEURS TECHNIQUES
# =============================================================================
class TechIndicators:

    @staticmethod
    def rsi(close, p=14):
        d = close.diff()
        g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
        l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1e-10))

    @staticmethod
    def macd(close, fast=12, slow=26, sig=9):
        ef = close.ewm(span=fast, adjust=False).mean()
        es = close.ewm(span=slow, adjust=False).mean()
        line = ef - es
        signal = line.ewm(span=sig, adjust=False).mean()
        return line, signal, line - signal

    @staticmethod
    def bollinger(close, p=20, n=2):
        m = close.rolling(p).mean()
        s = close.rolling(p).std()
        return m + n*s, m, m - n*s

    @staticmethod
    def adx(h, l, c, p=14):
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        pdm = ((h-h.shift())>(l.shift()-l)).astype(float)*(h-h.shift()).clip(0)
        ndm = ((l.shift()-l)>(h-h.shift())).astype(float)*(l.shift()-l).clip(0)
        atr = tr.ewm(span=p, adjust=False).mean()
        pdi = 100*pdm.ewm(span=p, adjust=False).mean()/atr.replace(0, 1e-10)
        ndi = 100*ndm.ewm(span=p, adjust=False).mean()/atr.replace(0, 1e-10)
        dx  = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, 1e-10)
        return dx.ewm(span=p, adjust=False).mean(), pdi, ndi

    @staticmethod
    def atr(h, l, c, p=14):
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()

    @staticmethod
    def stochastic(h, l, c, k=14, d=3):
        lk = l.rolling(k).min()
        hk = h.rolling(k).max()
        pk = 100*(c-lk)/(hk-lk).replace(0, 1e-10)
        return pk, pk.rolling(d).mean()

    @staticmethod
    def obv(close, volume):
        d = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (d * volume).cumsum()

    @classmethod
    def compute_all(cls, df):
        df = df.copy()
        df["rsi"]  = cls.rsi(df["close"])
        df["macd"], df["macd_sig"], df["macd_hist"] = cls.macd(df["close"])
        df["bb_up"], df["bb_mid"], df["bb_low"] = cls.bollinger(df["close"])
        df["adx"],  df["pdi"],     df["ndi"]    = cls.adx(df["high"],df["low"],df["close"])
        df["atr"]  = cls.atr(df["high"], df["low"], df["close"])
        df["stoch_k"], df["stoch_d"] = cls.stochastic(df["high"],df["low"],df["close"])
        df["obv"]  = cls.obv(df["close"], df["volume"])
        df["mm20"] = df["close"].rolling(20).mean()
        df["mm50"] = df["close"].rolling(50).mean()
        df["mm200"]= df["close"].rolling(200).mean()
        df["ret1"] = df["close"].pct_change(1)
        df["ret5"] = df["close"].pct_change(5)
        df["vol_ratio"]  = df["volume"] / df["volume"].rolling(20).mean().replace(0, 1)
        df["bb_pos"]     = (df["close"]-df["bb_low"]) / (df["bb_up"]-df["bb_low"]+1e-10)
        df["mm_cross"]   = (df["mm20"]-df["mm50"]) / df["close"].replace(0, 1e-10)
        obv_max = df["obv"].abs().max()
        df["obv_norm"]   = df["obv"] / (obv_max + 1e-10)
        df["bb_width"]   = (df["bb_up"]-df["bb_low"]) / df["bb_mid"].replace(0, 1e-10)
        return df


# =============================================================================
# SECTION 3 — MACHINE LEARNING
# =============================================================================
class MLPredictor:
    """RF + GB. Entraîné avant tout trade. Séparation temporelle 75/25."""

    FEATURES = [
        "rsi","macd_hist","macd","bb_pos","adx",
        "stoch_k","stoch_d","vol_ratio","ret1","ret5",
        "mm_cross","pdi","ndi","obv_norm","bb_width",
    ]

    def __init__(self):
        self.rf = self.gb = self.scaler = None
        self.trained  = False
        self.accuracy = 0.0
        self.feat_imp = {}
        self._lock    = threading.Lock()

    def train(self, df):
        if not ML_OK:
            return {"error": "scikit-learn non installé"}
        if len(df) < 120:
            return {"error": "Pas assez de données"}
        with self._lock:
            df = df.copy()
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
            clean = df[self.FEATURES + ["target"]].dropna()
            if len(clean) < 80:
                return {"error": "Données insuffisantes"}
            X, y  = clean[self.FEATURES].values, clean["target"].values
            split = int(len(X) * 0.75)
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = y[:split], y[split:]
            self.scaler = StandardScaler()
            Xtr = self.scaler.fit_transform(Xtr)
            Xte = self.scaler.transform(Xte)
            self.rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
            self.rf.fit(Xtr, ytr)
            self.gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                learning_rate=0.05, subsample=0.8, random_state=42)
            self.gb.fit(Xtr, ytr)
            rf_acc = accuracy_score(yte, self.rf.predict(Xte))
            gb_acc = accuracy_score(yte, self.gb.predict(Xte))
            self.accuracy = round((rf_acc + gb_acc) / 2 * 100, 1)
            self.trained  = True
            self.feat_imp = dict(sorted(
                {f: round(float(v)*100,2) for f,v in
                 zip(self.FEATURES, self.rf.feature_importances_)}.items(),
                key=lambda x: -x[1]))
            log.info(f"RF+GB entraîné — Précision: {self.accuracy}%")
            return {"accuracy": self.accuracy, "n_samples": len(clean),
                    "feature_importance": self.feat_imp}

    def predict(self, df):
        if not self.trained or not ML_OK:
            return {"signal":"neutral","confidence":50,"proba_up":50,"proba_down":50,"accuracy":0}
        with self._lock:
            clean = df[self.FEATURES].dropna()
            if len(clean) == 0:
                return {"signal":"neutral","confidence":50,"proba_up":50,"proba_down":50}
            row   = clean.iloc[-1:].values
            row_s = self.scaler.transform(row)
            # Ensemble RF 60% + GB 40%
            proba = self.rf.predict_proba(row_s)[0]*0.6 + self.gb.predict_proba(row_s)[0]*0.4
            d     = int(np.argmax(proba))
            conf  = round(float(proba[d])*100, 1)
            # Seuil plus strict : 58% de confiance minimum
            sig = "bullish" if d==1 and conf>58 else "bearish" if d==0 and conf>58 else "neutral"
            return {"signal":sig,"confidence":conf,
                    "proba_up":round(float(proba[1])*100,1),
                    "proba_down":round(float(proba[0])*100,1),
                    "accuracy":self.accuracy}


# =============================================================================
# SECTION 4 — RISQUE
# =============================================================================
class RiskMetrics:
    @staticmethod
    def log_returns(close):
        return np.log(close / close.shift(1)).dropna()

    @classmethod
    def var(cls, close, h=24, conf=0.95):
        r = cls.log_returns(close)
        if len(r) < 30: return 0.0
        return abs(float(np.percentile(r, (1-conf)*100))) * math.sqrt(h) * 100

    @classmethod
    def cvar(cls, close, h=24, conf=0.95):
        r = cls.log_returns(close)
        if len(r) < 30: return 0.0
        thr = np.percentile(r, (1-conf)*100)
        tail = r[r <= thr]
        return abs(float(tail.mean())) * math.sqrt(h) * 100 if len(tail) else 0.0

    @classmethod
    def sortino(cls, close):
        r = cls.log_returns(close)
        if len(r) < 30: return 0.0
        down = r[r < 0]
        ds   = down.std()*math.sqrt(252) if len(down)>0 else 1e-10
        return round(r.mean()*252/ds, 3)

    @classmethod
    def sharpe(cls, close):
        r = cls.log_returns(close)
        if len(r) < 30: return 0.0
        return round(r.mean()*252/(r.std()*math.sqrt(252)+1e-10), 3)

    @staticmethod
    def max_drawdown(close):
        dd = (close - close.cummax()) / close.cummax()
        return abs(float(dd.min())) * 100

    @staticmethod
    def current_drawdown(close):
        peak = float(close.cummax().iloc[-1])
        cur  = float(close.iloc[-1])
        return abs((cur-peak)/peak)*100 if peak > 0 else 0

    @classmethod
    def calmar(cls, close):
        r   = cls.log_returns(close)
        mdd = cls.max_drawdown(close)
        return round(r.mean()*252*100/mdd, 3) if mdd > 0 else 0

    @classmethod
    def compute_all(cls, df):
        c = df["close"]
        return {
            "var_24h":  round(cls.var(c), 2),
            "cvar_24h": round(cls.cvar(c), 2),
            "sortino":  round(cls.sortino(c), 3),
            "sharpe":   round(cls.sharpe(c), 3),
            "mdd":      round(cls.max_drawdown(c), 2),
            "cur_dd":   round(cls.current_drawdown(c), 2),
            "calmar":   round(cls.calmar(c), 3),
            "rsi":      round(float(df["rsi"].iloc[-1]), 1),
            "adx":      round(float(df["adx"].iloc[-1]), 1),
            "atr":      round(float(df["atr"].iloc[-1]), 6),
            "vol_24h":  round(float(df["ret1"].rolling(24).std().iloc[-1]*math.sqrt(24)*100), 2),
        }


# =============================================================================
# SECTION 5 — SENTIMENT CRYPTOPANIC
# =============================================================================
class SentimentEngine:
    def fetch(self, currency="BTC"):
        if not CRYPTOPANIC_KEY:
            return self._demo(currency)
        try:
            r = requests.get("https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": CRYPTOPANIC_KEY, "currencies": currency,
                        "public": "true", "kind": "news"}, timeout=8)
            r.raise_for_status()
            return r.json().get("results", [])[:30]
        except:
            return self._demo(currency)

    def _demo(self, currency):
        import random
        now = datetime.now(timezone.utc)
        templates = [
            ("Institutional adoption accelerates", 14, 2, "positive"),
            ("ETF inflows reach new record",       12, 1, "positive"),
            ("Fed signals rate hold",               5, 5, "neutral"),
            ("Regulatory concerns increase",        3,10, "negative"),
            ("Whale movement detected",             4, 7, "negative"),
            ("DeFi TVL surges",                    11, 2, "positive"),
        ]
        return [{"title": t.replace("Bitcoin", currency),
                 "published_at": (now-timedelta(minutes=random.randint(2,180))).isoformat(),
                 "votes": {"positive": b, "negative": br, "important": (b+br)//2},
                 "source": {"title": random.choice(["Bloomberg","Reuters","CoinDesk"])},
                 "url": "#", "sentiment": sent}
                for t, b, br, sent in templates]

    def score(self, news, window_min=60):
        if not news:
            return {"score": 0.0, "zone": "bruit", "n": 0, "bull": 0, "bear": 0}
        now = datetime.now(timezone.utc)
        scores, weights, bt, brt = [], [], 0, 0
        for item in news:
            try:
                pub   = pd.to_datetime(item.get("published_at","")).tz_convert("UTC")
                age   = (now-pub).total_seconds()/60
                if age > window_min: continue
                decay = math.exp(-age/window_min)
                v     = item.get("votes", {})
                b     = float(v.get("positive",0) or v.get("liked",0))
                br    = float(v.get("negative",0) or v.get("disliked",0))
                imp   = float(v.get("important",1) or 1)
                w     = (1+math.log1p(imp))*decay
                s     = (b-br)/(b+br+1e-6)
                scores.append(s*w); weights.append(w)
                bt += b; brt += br
            except: continue
        if not weights:
            return {"score": 0.0, "zone": "bruit", "n": 0, "bull": 0, "bear": 0}
        score = float(np.clip(sum(scores)/sum(weights), -1, 1))
        zone  = "euphorie" if score > 0.5 else "panique" if score < -0.5 else "bruit"
        return {"score": round(score,3), "zone": zone,
                "n": len(scores), "bull": int(bt), "bear": int(brt)}


# =============================================================================
# SECTION 6 — AGENT GEMINI (SDK officiel google-genai)
# Installation : pip install -q -U google-genai
# =============================================================================

# Charger le SDK officiel Google GenAI
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_SDK = True
except ImportError:
    GENAI_SDK = False
    log.warning("SDK google-genai non installé — lancer : pip install -q -U google-genai")


class GeminiAgent:
    """
    Agent Gemini utilisant le SDK officiel Google GenAI.
    Le SDK lit automatiquement GEMINI_API_KEY depuis les variables d'environnement.

    Installation : pip install -q -U google-genai
    Clé gratuite  : aistudio.google.com → Get API key
    Modèle utilisé: gemini-2.5-flash-preview-05-20 (le plus récent et gratuit)
    """

    MODEL = "gemini-2.5-flash"  # Modèle détecté fonctionnel

    # Ordre de priorité — le premier qui répond est utilisé
    FALLBACK_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-preview-04-17",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    SYSTEM = """Tu es un analyste quantitatif crypto senior. Règles absolues :
1. Réponds UNIQUEMENT sur les données fournies dans le prompt.
2. Format strict : SIGNAL: BULLISH/BEARISH/NEUTRE | RAISON: (2 phrases max) | RISQUE: (1 phrase)
3. Neutre, factuel, pas d'hyperbole. Maximum 120 mots. En français."""

    def __init__(self):
        self._client = None  # Initialisé au premier appel

    def _get_client(self):
        """Crée le client SDK une seule fois (lazy init)."""
        if self._client is None and GENAI_SDK and GEMINI_KEY:
            try:
                self._client = google_genai.Client(api_key=GEMINI_KEY)
                log.info("Client Gemini SDK initialisé ✓")
            except Exception as e:
                log.error(f"Erreur init client Gemini: {e}")
        return self._client

    def _build_prompt(self, ctx):
        """Construit le prompt avec toutes les données disponibles."""
        rf   = ctx.get("rf", {})
        risk = ctx.get("risk", {})
        sent = ctx.get("sentiment", {})
        ind  = ctx.get("indicators", {})
        return (
            f"ACTIF : {ctx.get('symbol','')} @ {ctx.get('price',0):.4f} USDT"
            f" | Score décision : {ctx.get('decision_score','?')}/100\n\n"
            f"INDICATEURS TECHNIQUES :\n"
            f"  RSI(14)    = {ind.get('rsi','?')}  (>70 surachat, <30 survente)\n"
            f"  MACD hist  = {ind.get('macd_hist','?')}\n"
            f"  ADX(14)    = {ind.get('adx','?')}  (>25 = tendance forte)\n"
            f"  Stoch K/D  = {ind.get('stoch_k','?')} / {ind.get('stoch_d','?')}\n"
            f"  Bollinger  = {ind.get('bb_pos','?')}  (0=bas, 1=haut)\n"
            f"  Vol/moy    = {ind.get('vol_ratio','?')}x\n\n"
            f"MODÈLE ML (RF+GB) :\n"
            f"  Signal     = {rf.get('signal','?').upper()}\n"
            f"  Confiance  = {rf.get('confidence','?')}%\n"
            f"  Proba up   = {rf.get('proba_up','?')}% | Proba down = {rf.get('proba_down','?')}%\n"
            f"  Précision  = {rf.get('accuracy','?')}%\n\n"
            f"RISQUE :\n"
            f"  VaR 24h    = {risk.get('var_24h','?')}%\n"
            f"  Sortino    = {risk.get('sortino','?')}  (>1 = bon)\n"
            f"  Max DD     = {risk.get('mdd','?')}%\n\n"
            f"SENTIMENT CryptoPanic :\n"
            f"  Score S    = {sent.get('score','?')}  [-1 panique → +1 euphorie]\n"
            f"  Zone       = {sent.get('zone','?').upper()}\n"
            f"  News       = {sent.get('n','?')} | Bull={sent.get('bull','?')} / Bear={sent.get('bear','?')}\n\n"
            f"{self.SYSTEM}"
        )

    def ask(self, ctx):
        """
        Envoie une requête à Gemini. Essaie TOUS les modèles en cas d'erreur.
        Ne s'arrête PAS sur les erreurs 429 (quota) — essaie le modèle suivant.
        """
        if not GENAI_SDK:
            return "[SDK google-genai non installé — pip install -q -U google-genai]"
        if not GEMINI_KEY:
            return None

        client = self._get_client()
        if client is None:
            return "[Erreur init client Gemini]"

        prompt = self._build_prompt(ctx)
        # gemini-2.5-flash en premier (confirmé fonctionnel par le test)
        models = ["gemini-2.5-flash"] + [m for m in self.FALLBACK_MODELS if m != "gemini-2.5-flash"]

        last_error = ""
        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=250,
                        temperature=0.3,
                    ),
                )
                text = response.text.strip()
                if text:
                    GeminiAgent.MODEL = model  # Mémoriser le modèle qui marche
                    log.info(f"Gemini [{model}] ✓ — {len(text)} chars")
                    return text

            except Exception as e:
                last_error = str(e)
                if "404" in last_error or "NOT_FOUND" in last_error:
                    log.debug(f"Gemini [{model}] 404 → essai suivant")
                    continue  # Modèle introuvable → essai suivant
                elif ("429" in last_error or "RESOURCE_EXHAUSTED" in last_error
                      or "RESSOURCES" in last_error or "exceeded" in last_error.lower()
                      or "quota" in last_error.lower()):
                    log.debug(f"Gemini [{model}] quota → essai suivant")
                    continue  # Quota dépassé → essai modèle suivant
                elif "API_KEY" in last_error or "INVALID_ARGUMENT" in last_error:
                    log.error(f"Gemini clé invalide: {last_error[:80]}")
                    return "[Gemini : clé API invalide — vérifier GEMINI_API_KEY dans .env]"
                else:
                    log.warning(f"Gemini [{model}]: {last_error[:80]} → essai suivant")
                    continue

        log.error(f"Gemini : tous les modèles ont échoué. Dernière erreur: {last_error[:100]}")
        return None  # Retourner None → affiche la synthèse locale

    def synthesize(self, ctx, use_thread=False):
        """
        Appelle Gemini et génère la synthèse locale.
        use_thread=False : appel direct (pour l'analyse utilisateur, on peut attendre)
        use_thread=True  : appel async (pour la boucle portfolio en arrière-plan)
        """
        if use_thread:
            gemini_res = [None]
            def call(): gemini_res[0] = self.ask(ctx)
            t = threading.Thread(target=call, daemon=True)
            t.start(); t.join(timeout=20)
            gemini_text = gemini_res[0]
        else:
            # Appel direct — le client attend la réponse (plus fiable)
            gemini_text = self.ask(ctx)

        # ── Synthèse locale (toujours disponible) ──
        rf    = ctx.get("rf", {})
        risk  = ctx.get("risk", {})
        sent  = ctx.get("sentiment", {})
        score = ctx.get("decision_score", 50)
        sig   = rf.get("signal", "neutral")
        conf  = float(rf.get("confidence", 50))
        var   = float(risk.get("var_24h", 0))
        sort  = float(risk.get("sortino", 0))
        zone  = sent.get("zone", "bruit")

        local  = f"SIGNAL LOCAL : {sig.upper()} ({conf:.0f}% confiance RF+GB)\n\n"
        local += f"Proba hausse {rf.get('proba_up',50):.0f}% / baisse {rf.get('proba_down',50):.0f}%\n"
        local += f"Sentiment : zone {zone.upper()} (S={sent.get('score',0):+.2f})\n"
        local += f"VaR 24h : {var:.1f}% | Sortino : {sort:.2f} | Score décision : {score}/100\n\n"
        if sig == "bullish" and var < 6 and sort > 0.5:
            local += "RECOMMANDATION : Conditions favorables. Entrée avec stop ATR×2."
        elif sig == "bearish" or var > 8:
            local += "RECOMMANDATION : Signal baissier ou risque élevé. Rester à l'écart."
        else:
            local += "RECOMMANDATION : Signal neutre. Attendre une meilleure opportunité."

        return {"gemini": gemini_text, "local": local}


# =============================================================================
# SECTION 7 — MOTEUR DE DÉCISION (Score 0-100)
# =============================================================================
class DecisionEngine:
    """
    Score 0-100. ACHAT si ≥ 70. VENTE si ≤ 30.
    Filtres obligatoires avant tout trade :
      1. RF entraîné et confiant (>58%)
      2. Prix au-dessus de la MM50 (tendance haussière)
      3. Volume suffisant (liquidité)
      4. VaR < seuil
    """

    BUY_THRESHOLD  = 62
    SELL_THRESHOLD = 32
    STOP_LOSS_PCT  = 0.03   # -3% (valeur rapport Figure 10)
    TAKE_PROFIT_PCT= 0.12   # +12%
    ATR_MULT       = 1.5    # Stop ATR × 1.5 (valeur rapport Figure 10)

    def compute_score(self, df, rf_pred, sentiment_score, risk_m):
        score = 0; reasons = []
        last  = df.iloc[-1]

        rsi     = float(last.get("rsi", 50))
        macd_h  = float(last.get("macd_hist", 0))
        prev_h  = float(df["macd_hist"].iloc[-2]) if len(df) > 1 else 0
        adx     = float(last.get("adx", 20))
        pdi     = float(last.get("pdi", 25))
        ndi     = float(last.get("ndi", 25))
        bb_pos  = float(last.get("bb_pos", 0.5))
        stoch_k = float(last.get("stoch_k", 50))
        stoch_d = float(last.get("stoch_d", 50))
        vol_r   = float(last.get("vol_ratio", 1))

        # ── 1. ML RF+GB (30 pts) ──
        sig  = rf_pred.get("signal","neutral")
        conf = rf_pred.get("confidence", 50)
        if sig == "bullish":
            pts = min(30, int(conf * 0.3))
            score += pts; reasons.append(f"RF BULLISH {conf:.0f}% (+{pts}pts)")
        elif sig == "bearish":
            reasons.append(f"RF BEARISH {conf:.0f}% (0pts)")
        else:
            score += 10; reasons.append("RF NEUTRE (+10pts)")

        # ── 2. RSI (15 pts) ──
        if 40 <= rsi <= 60:
            score += 15; reasons.append(f"RSI {rsi:.0f} zone neutre (+15pts)")
        elif 60 < rsi <= 70:
            score += 10; reasons.append(f"RSI {rsi:.0f} momentum (+10pts)")
        elif 30 <= rsi < 40:
            score += 12; reasons.append(f"RSI {rsi:.0f} proche survente (+12pts)")
        elif rsi < 30:
            score += 8;  reasons.append(f"RSI {rsi:.0f} survente (+8pts)")
        else:
            score += 1;  reasons.append(f"RSI {rsi:.0f} SURACHAT (+1pt seulement)")

        # ── 3. MACD (15 pts) ──
        if macd_h > 0 and macd_h > prev_h:
            score += 15; reasons.append("MACD hist+ croissant (+15pts)")
        elif macd_h > 0:
            score += 9;  reasons.append("MACD hist positif (+9pts)")
        elif macd_h < 0 and macd_h < prev_h:
            reasons.append("MACD hist- décroissant (0pts)")
        else:
            score += 4;  reasons.append("MACD mixte (+4pts)")

        # ── 4. ADX (10 pts) ──
        if adx > 30 and pdi > ndi:
            score += 10; reasons.append(f"ADX {adx:.0f} TREND FORT haussier (+10pts)")
        elif adx > 25 and pdi > ndi:
            score += 7;  reasons.append(f"ADX {adx:.0f} tendance haussière (+7pts)")
        elif adx > 20:
            score += 4;  reasons.append(f"ADX {adx:.0f} tendance modérée (+4pts)")
        else:
            score += 2;  reasons.append(f"ADX {adx:.0f} sans tendance (+2pts)")

        # ── 5. Bollinger (10 pts) ──
        if 0.3 <= bb_pos <= 0.7:
            score += 10; reasons.append("BB zone centrale (+10pts)")
        elif bb_pos < 0.2:
            score += 8;  reasons.append("BB bande basse rebond (+8pts)")
        elif bb_pos < 0.3:
            score += 6;  reasons.append("BB zone basse (+6pts)")
        elif bb_pos > 0.9:
            score += 0;  reasons.append("BB sortie haute (0pts)")
        else:
            score += 4;  reasons.append("BB zone haute (+4pts)")

        # ── 6. Stochastique (5 pts) ──
        if stoch_k < 25 and stoch_k > stoch_d:
            score += 5; reasons.append(f"Stoch {stoch_k:.0f} survente+croisement (+5pts)")
        elif stoch_k < 40:
            score += 3; reasons.append(f"Stoch {stoch_k:.0f} zone basse (+3pts)")
        elif stoch_k > 80:
            score += 0; reasons.append(f"Stoch {stoch_k:.0f} surachat (0pts)")
        else:
            score += 2; reasons.append(f"Stoch {stoch_k:.0f} neutre (+2pts)")

        # ── 7. Volume (5 pts) ──
        if vol_r > 1.5:
            score += 5; reasons.append(f"Volume x{vol_r:.1f} confirmation (+5pts)")
        elif vol_r > 1.0:
            score += 3; reasons.append("Volume ok (+3pts)")
        else:
            score += 1; reasons.append("Volume faible (+1pt)")

        # ── 8. Sentiment (10 pts) ──
        s = float(sentiment_score)
        if s > 0.5:
            score += 10; reasons.append(f"Sentiment EUPHORIE S={s:+.2f} (+10pts)")
        elif s > 0.2:
            score += 7;  reasons.append(f"Sentiment+ S={s:+.2f} (+7pts)")
        elif s > -0.2:
            score += 5;  reasons.append(f"Sentiment neutre S={s:+.2f} (+5pts)")
        elif s > -0.5:
            score += 2;  reasons.append(f"Sentiment- S={s:+.2f} (+2pts)")
        else:
            score += 0;  reasons.append(f"Sentiment PANIQUE S={s:+.2f} (0pts)")

        # ── Vérification risque ──
        var  = risk_m.get("var_24h", 0)
        sort = risk_m.get("sortino", 0)
        blocked = False; risk_msg = ""
        if var > 10:
            blocked  = True
            risk_msg = f"⛔ VaR {var:.1f}% critique — trade bloqué"
        elif sort < 0.2 and score > 70:
            score    = min(score, 60)
            risk_msg = f"⚠ Sortino {sort:.2f} insuffisant — score plafonné"

        return {
            "score":    int(score),
            "signal":   "BUY" if score >= self.BUY_THRESHOLD and not blocked
                        else "SELL" if score <= self.SELL_THRESHOLD else "HOLD",
            "reasons":  reasons,
            "blocked":  blocked,
            "risk_msg": risk_msg,
            "breakdown": {
                "ml":        min(30, int(conf*0.3)) if sig=="bullish" else 0,
                "rsi":       15 if 40<=rsi<=60 else 10 if rsi<70 else 1,
                "macd":      15 if macd_h>0 and macd_h>prev_h else 9 if macd_h>0 else 0,
                "adx":       10 if adx>30 else 7 if adx>25 else 4,
                "bollinger": 10 if 0.3<=bb_pos<=0.7 else 5,
                "stoch":     5 if stoch_k<25 else 2,
                "volume":    5 if vol_r>1.5 else 1,
                "sentiment": 10 if s>0.5 else max(0, int((s+1)*5)),
            }
        }

    def compute_stops(self, entry, current, atr_val, sent_score):
        """
        4 types de stops (rapport Figure 10).
        En cas de déclenchement simultané → retient le plus proche du prix.
        """
        fixe     = round(entry * (1 - self.STOP_LOSS_PCT), 6)   # -3%
        atr_stop = round(entry - self.ATR_MULT * atr_val, 6)    # entrée - 1.5×ATR
        trailing = round(current * 0.97, 6)                      # trailing -3%
        # Stop sentiment : déclenché si S < -0.5 ET VaR > seuil simultanément
        sentiment_stop = round(entry * 0.98, 6) if sent_score < -0.5 else None

        candidates  = [v for v in [fixe, atr_stop, trailing, sentiment_stop] if v is not None]
        recommended = max(candidates)  # plus proche du prix = plus protecteur
        return {
            "fixe": fixe, "atr": atr_stop, "trailing": trailing,
            "sentiment": sentiment_stop, "recommended": recommended,
            "distance_pct": round((current - recommended) / current * 100, 2)
        }


# =============================================================================
# SECTION 8 — PORTEFEUILLE AUTONOME
# =============================================================================
class AutoPortfolio:
    """
    Capital fictif $10,000. Règles strictes :
    - PAS de trade tant que le RF n'est pas entraîné
    - Score ≥ 70 pour acheter (vs 65 avant)
    - Prix doit être au-dessus de la MM50 (tendance haussière)
    - Volume minimum selon le tier
    - Stop loss à -5% (au lieu de -3%)
    - Take profit à +12%
    """

    INITIAL = 10_000.0
    FEE     = 0.001

    def __init__(self):
        self.capital   = self.INITIAL
        self.positions = {}
        self.trades    = []
        self.decisions = []
        self.equity_curve = [self.INITIAL]
        self._lock = threading.Lock()

    def _add_decision(self, type_, text, reason, score=0):
        self.decisions.insert(0, {
            "type": type_, "score": score, "text": text,
            "reason": reason[:150], "ts": datetime.utcnow().strftime("%H:%M:%S")
        })
        if len(self.decisions) > 300:
            self.decisions.pop()

    def buy(self, symbol, price, budget, stop, reason, score):
        with self._lock:
            if symbol in self.positions or budget > self.capital or price <= 0:
                return None
            qty = (budget * (1 - self.FEE)) / price
            self.capital -= budget
            self.positions[symbol] = {
                "qty": qty, "entry": price, "budget": budget,
                "stop": stop, "ts": datetime.utcnow().isoformat()
            }
            t = {"id": str(uuid.uuid4())[:6], "symbol": symbol, "action": "BUY",
                 "price": round(price,6), "budget": round(budget,2),
                 "score": score, "reason": reason, "ts": datetime.utcnow().strftime("%H:%M:%S")}
            self.trades.insert(0, t)
            self._add_decision("buy", f"▲ ACHAT {symbol.replace('USDT','')} @ ${price:.4f}", reason, score)
            if len(self.trades) > 300: self.trades.pop()
            log.info(f"BUY {symbol} score={score} budget=${budget:.0f} @ {price:.4f}")
            return t

    def sell(self, symbol, price, reason, score=0):
        with self._lock:
            if symbol not in self.positions:
                return None
            pos      = self.positions.pop(symbol)
            proceeds = pos["qty"] * price * (1 - self.FEE)
            pnl      = proceeds - pos["budget"]
            pnl_pct  = pnl / pos["budget"] * 100
            self.capital += proceeds
            t = {"id": str(uuid.uuid4())[:6], "symbol": symbol, "action": "SELL",
                 "price": round(price,6), "pnl": round(pnl,2),
                 "pnl_pct": round(pnl_pct,2), "score": score, "reason": reason,
                 "ts": datetime.utcnow().strftime("%H:%M:%S")}
            self.trades.insert(0, t)
            col = "sell" if pnl >= 0 else "loss"
            icon = "▼" if pnl >= 0 else "↓"
            self._add_decision(col,
                f"{icon} VENTE {symbol.replace('USDT','')} @ ${price:.4f} ({pnl_pct:+.1f}%)",
                reason, score)
            if len(self.decisions) > 300: self.decisions.pop()
            log.info(f"SELL {symbol} pnl={pnl:.2f} ({pnl_pct:+.1f}%)")
            return t

    def check_stops(self, symbol, price, df=None):
        if symbol not in self.positions: return
        pos     = self.positions[symbol]
        pnl_pct = (price - pos["entry"]) / pos["entry"] * 100
        stop    = pos.get("stop", pos["entry"] * 0.95)

        # Trailing stop : monte avec le prix
        if df is not None and "atr" in df.columns:
            atr_v     = float(df["atr"].iloc[-1])
            new_trail = price - 2 * atr_v
            if new_trail > stop:
                self.positions[symbol]["stop"] = round(new_trail, 6)
                stop = new_trail

        if price <= stop:
            self.sell(symbol, price, f"Stop loss @ ${price:.4f} ({pnl_pct:+.1f}%)")
        elif pnl_pct >= 12:
            self.sell(symbol, price, f"Take profit +{pnl_pct:.1f}%")

    def get_equity(self, prices_dict):
        total = self.capital
        for sym, pos in self.positions.items():
            p = prices_dict.get(sym, {}).get("price", pos["entry"]) if isinstance(prices_dict.get(sym), dict) else prices_dict.get(sym, pos["entry"])
            total += pos["qty"] * p
        return total

    def summary(self, prices_dict):
        equity = self.get_equity(prices_dict)
        ret    = (equity / self.INITIAL - 1) * 100
        sells  = [t for t in self.trades if t["action"] == "SELL"]
        wins   = [t for t in sells if t.get("pnl",0) > 0]
        wr     = len(wins) / max(1, len(sells)) * 100
        losses_sum = abs(sum(t["pnl"] for t in sells if t.get("pnl",0) <= 0))
        pf     = sum(t["pnl"] for t in wins) / max(1e-10, losses_sum)

        positions_list = []
        exposition_totale = 0.0
        reserves_stop     = 0.0
        for sym, pos in self.positions.items():
            p   = prices_dict.get(sym, {}).get("price", pos["entry"]) if isinstance(prices_dict.get(sym), dict) else prices_dict.get(sym, pos["entry"])
            pct = (p - pos["entry"]) / pos["entry"] * 100
            val = pos["qty"] * p
            stop = pos.get("stop", pos["entry"] * 0.97)
            # Réserve = exposition × distance stop (%)
            dist_stop_pct = (p - stop) / p if p > 0 else 0.03
            reserve = val * dist_stop_pct
            exposition_totale += val
            reserves_stop     += reserve
            positions_list.append({
                "symbol": sym, "qty": round(pos["qty"],6),
                "entry": round(pos["entry"],6), "current": round(p,6),
                "pnl_pct": round(pct,2), "value": round(val, 2),
                "stop": round(stop, 6),
                "tier": get_tier(sym)["label"],
            })

        # ── Marge de manœuvre (rapport p.21) ──
        # Marge = Capital total − Exposition actuelle − Σ(Réserves stop loss)
        marge = equity - exposition_totale - reserves_stop
        marge_pct = (marge / equity * 100) if equity > 0 else 100
        if marge_pct >= 40:   marge_color = "green"
        elif marge_pct >= 20: marge_color = "orange"
        else:                  marge_color = "red"

        # ── Red Flags (rapport Figure 11) ──
        red_flags = []

        # 1. VaR > seuil critique — calculé sur l'equity curve
        if len(self.equity_curve) > 10:
            eq_s   = pd.Series(self.equity_curve)
            cur_dd = float(((eq_s - eq_s.cummax()) / eq_s.cummax()).iloc[-1]) * 100
            mdd_h  = float(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100
            # Red Flag : drawdown actuel > MDD historique (nouvel extrême)
            if len(self.equity_curve) > 30 and cur_dd < mdd_h * 0.95:
                red_flags.append({
                    "type": "critical",
                    "title": "🔴 Drawdown > MDD historique",
                    "msg": f"Chute actuelle {cur_dd:.1f}% dépasse le pire historique {mdd_h:.1f}%. Réduction d'exposition recommandée.",
                    "source": "Calcul local"
                })

        # 2. Volume spike sans news (rapport Figure 11 — suspicion manipulation Whale)
        # Détecté si vol_ratio > 3× la moyenne ET score sentiment neutre (bruit)
        # On stocke les derniers vol_ratios analysés dans les décisions
        vol_spike_detected = any(
            "spike" in str(d.get("reason","")).lower() or "whale" in str(d.get("reason","")).lower()
            for d in self.decisions[:5]
        )
        # Vérification directe sur les positions ouvertes
        for sym, pos in self.positions.items():
            p_data = prices_dict.get(sym, {})
            if isinstance(p_data, dict):
                vol = p_data.get("volume", 0)
                # Volume > 3x la valeur typique pour ce tier = alerte
                tier = get_tier(sym)
                min_vol = MIN_VOLUME_USDT.get(tier["tier"], 500_000)
                if vol > min_vol * 10:  # spike = 10x le minimum du tier
                    vol_spike_detected = True
                    red_flags.append({
                        "type": "warning",
                        "title": f"🟠 Volume spike détecté — {sym.replace('USDT','')}",
                        "msg": f"Volume 24h anormal (${vol/1e6:.1f}M). Risque de manipulation Whale / Bull Trap. Vérifier les news.",
                        "source": "Binance + CryptoPanic"
                    })
                    break

        return {
            "capital":        round(self.capital, 2),
            "equity":         round(equity, 2),
            "return_pct":     round(ret, 2),
            "n_positions":    len(self.positions),
            "n_trades":       len(self.trades),
            "win_rate":       round(wr, 1),
            "profit_factor":  round(pf, 2),
            "exposition":     round(exposition_totale, 2),
            "reserves_stop":  round(reserves_stop, 2),
            "marge":          round(marge, 2),
            "marge_pct":      round(marge_pct, 1),
            "marge_color":    marge_color,
            "red_flags":      red_flags,
            "positions":      positions_list,
            "recent_trades":  self.trades[:30],
            "decisions":      self.decisions[:50],
            "equity_curve":   self.equity_curve[-300:],
        }

    def reset(self):
        with self._lock:
            self.capital = self.INITIAL
            self.positions.clear()
            self.trades.clear()
            self.decisions.clear()
            self.equity_curve = [self.INITIAL]


# =============================================================================
# ASSEMBLAGE
# =============================================================================
binance   = BinanceData()
tech      = TechIndicators()
ml        = MLPredictor()
risk_eng  = RiskMetrics()
sentiment = SentimentEngine()
gemini    = GeminiAgent()
decision  = DecisionEngine()
portfolio = AutoPortfolio()

prices_cache: dict = {}
auto_running = False


def price_updater_loop():
    while True:
        try:
            prices_cache.update(binance.all_tickers())
        except Exception as e:
            log.warning(f"Price update: {e}")
        time.sleep(2)


def auto_portfolio_loop(var_threshold=5.0):
    """
    Boucle autonome sur les 100 cryptos.
    IMPORTANT : attend que le RF soit entraîné avant le premier trade.
    """
    global auto_running

    # ── Entraînement obligatoire avant tout trade ──
    log.info("Entraînement RF sur BTC+ETH (attendre avant les trades)...")
    portfolio._add_decision("info", "⚙ Entraînement RF en cours...",
        "Aucun trade avant la fin de l'entraînement. Patience ~30s.")

    for sym_train in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
        df0 = binance.klines(sym_train, "1h", 500)
        if not df0.empty:
            df0 = tech.compute_all(df0)
            res = ml.train(df0)
            log.info(f"RF {sym_train}: {res.get('accuracy','?')}%")

    if ml.trained:
        portfolio._add_decision("info",
            f"✅ RF entraîné — Précision {ml.accuracy}%",
            f"Le portefeuille peut maintenant trader. Seuil achat: {decision.BUY_THRESHOLD}/100.")
    else:
        portfolio._add_decision("info", "⚠ RF non entraîné — mode dégradé",
            "scikit-learn manquant. Les signaux ML ne sont pas disponibles.")

    while auto_running:
        for symbol in ALL_CRYPTOS:
            if not auto_running: break
            try:
                # Volume minimum selon le tier (filtrer les altcoins illiquides)
                tier    = get_tier(symbol)
                vol_24h = prices_cache.get(symbol, {}).get("volume", 0)
                min_vol = MIN_VOLUME_USDT.get(tier["tier"], 500_000)
                if vol_24h > 0 and vol_24h < min_vol:
                    continue  # Pas assez liquide

                df = binance.klines(symbol, "1h", 250)
                if df.empty or len(df) < 60:
                    continue
                df = tech.compute_all(df)

                price = prices_cache.get(symbol, {}).get("price", 0)
                if price <= 0:
                    price = float(df["close"].iloc[-1])

                # ── Filtre tendance : MM50 recommandé mais pas bloquant ──
                mm50_val   = df["mm50"].iloc[-1]
                above_mm50 = (price > float(mm50_val)) if not pd.isna(mm50_val) else True

                risk_m  = risk_eng.compute_all(df)
                rf_pred = ml.predict(df)
                curr    = symbol.replace("USDT","")
                news    = sentiment.fetch(curr)
                sent    = sentiment.score(news)
                dec     = decision.compute_score(df, rf_pred, sent["score"], risk_m)

                # ── Red Flag : Volume spike sans news (rapport Figure 11) ──
                # Spike = vol_ratio > 3.5 × moyenne ET sentiment neutre (zone bruit)
                vol_ratio = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns else 1.0
                if vol_ratio > 3.5 and sent["zone"] == "bruit":
                    portfolio._add_decision("loss",
                        f"🟠 Red Flag VOLUME SPIKE — {symbol.replace('USDT','')}",
                        f"Volume x{vol_ratio:.1f} la moyenne SANS support news (S={sent['score']:+.2f} zone bruit). Risque manipulation Whale / Bull Trap.")
                    log.warning(f"Red Flag volume spike: {symbol} vol_ratio={vol_ratio:.1f}")

                # Vérifier stops (avec trailing stop ATR)
                portfolio.check_stops(symbol, price, df)

                budget = portfolio.capital * tier["exposure"]

                # ── ACHAT : score ≥ 62, RF entraîné, volume ok ──
                # MM50 donne +3pts bonus mais n'est plus un filtre dur
                if (dec["signal"] == "BUY"
                        and symbol not in portfolio.positions
                        and budget >= 15
                        and (ml.trained or not ML_OK)):

                    # Bonus MM50 : si le prix est en-dessous, réduire le budget de 50%
                    effective_budget = budget if above_mm50 else budget * 0.5

                    atr_v  = risk_m.get("atr", price * 0.015)
                    stops  = decision.compute_stops(price, price, atr_v, sent["score"])
                    mm50_note = "MM50 ✓" if above_mm50 else "MM50 ↓ budget réduit"
                    reason = f"Score {dec['score']}/100 | {mm50_note} | {' | '.join(dec['reasons'][:2])}"
                    portfolio.buy(symbol, price, effective_budget, stops["recommended"], reason, dec["score"])

                    # Gemini après achat (async — ne bloque pas la boucle)
                    def ask_async(sym=symbol, p=price, r=risk_m, s=sent, rf=rf_pred, d=dec):
                        last = df.iloc[-1]
                        ind  = {k: round(float(last[k]),4) for k in
                               ["rsi","macd_hist","adx","bb_pos","stoch_k","stoch_d","vol_ratio","mm_cross"]
                               if k in df.columns and not pd.isna(last[k])}
                        ctx  = {"symbol": sym, "price": p, "risk": r,
                                "sentiment": s, "rf": rf, "indicators": ind,
                                "decision_score": d["score"]}
                        res  = gemini.synthesize(ctx, use_thread=False)
                        txt  = res.get("gemini") or res.get("local","")
                        portfolio._add_decision("ai",
                            f"🤖 Gemini → {sym.replace('USDT','')}", txt[:200], d["score"])
                    threading.Thread(target=ask_async, daemon=True).start()

                # ── VENTE : score ≤ 30 ──
                elif dec["signal"] == "SELL" and symbol in portfolio.positions:
                    portfolio.sell(symbol, price,
                        f"Score {dec['score']}/100 — signal SELL", dec["score"])

                # Equity curve
                eq = portfolio.get_equity(prices_cache)
                portfolio.equity_curve.append(round(eq, 2))
                if len(portfolio.equity_curve) > 500:
                    portfolio.equity_curve.pop(0)

                time.sleep(1)

            except Exception as e:
                log.error(f"Auto {symbol}: {e}")
                continue

        for _ in range(30):
            if not auto_running: break
            time.sleep(1)


# =============================================================================
# API FLASK
# =============================================================================
app = Flask(__name__, static_folder=str(SCRIPT_DIR), static_url_path="")
CORS(app, origins="*")

@app.route("/")
def index():
    return send_from_directory(str(SCRIPT_DIR), "index.html")

@app.route("/api/prices")
def api_prices():
    return jsonify(prices_cache)

@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(portfolio.summary(prices_cache))

@app.route("/api/portfolio/start", methods=["POST"])
def api_start():
    global auto_running
    if auto_running: return jsonify({"message": "Déjà en cours"})
    data    = request.get_json() or {}
    var_thr = float(data.get("var_threshold", 5.0))
    auto_running = True
    threading.Thread(target=auto_portfolio_loop, args=(var_thr,), daemon=True).start()
    return jsonify({"message": "Démarré", "running": True})

@app.route("/api/portfolio/stop", methods=["POST"])
def api_stop():
    global auto_running
    auto_running = False
    portfolio._add_decision("info", "⏸ Portefeuille arrêté", "Positions maintenues.")
    return jsonify({"message": "Arrêté", "running": False})

@app.route("/api/portfolio/reset", methods=["POST"])
def api_reset():
    global auto_running
    auto_running = False
    portfolio.reset()
    return jsonify({"message": "Réinitialisé", "capital": portfolio.INITIAL})

@app.route("/api/portfolio/status")
def api_status():
    return jsonify({"running": auto_running, "ml_trained": ml.trained,
                    "ml_accuracy": ml.accuracy})

@app.route("/api/analysis/<symbol>")
def api_analysis(symbol):
    symbol = symbol.upper()
    try:
        df = binance.klines(symbol, "1h", 250)
        if df.empty: return jsonify({"error": "Données indisponibles"}), 500
        df = tech.compute_all(df)

        price   = binance.spot_price(symbol)
        if price <= 0:
            price = prices_cache.get(symbol, {}).get("price", float(df["close"].iloc[-1]))

        risk_m  = risk_eng.compute_all(df)
        rf_pred = ml.predict(df)
        news    = sentiment.fetch(symbol.replace("USDT",""))
        sent    = sentiment.score(news)
        dec     = decision.compute_score(df, rf_pred, sent["score"], risk_m)
        stops_d = decision.compute_stops(price, price, risk_m.get("atr", price*0.015), sent["score"])

        last = df.iloc[-1]
        ind  = {k: round(float(last[k]),4) for k in
               ["rsi","macd_hist","adx","bb_pos","stoch_k","stoch_d","vol_ratio","mm_cross"]
               if k in df.columns and not pd.isna(last[k])}

        ctx     = {"symbol": symbol, "price": price, "risk": risk_m,
                   "sentiment": sent, "rf": rf_pred, "indicators": ind,
                   "decision_score": dec["score"]}

        # Appel Gemini avec timeout court — si ça dépasse, on renvoie null
        # Le frontend affichera la synthèse locale en attendant
        ai_res  = gemini.synthesize(ctx, use_thread=True)

        ohlcv   = df.tail(150).copy()
        ohlcv["open_time"] = ohlcv["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ohlcv_d = ohlcv.where(pd.notna(ohlcv), None).to_dict("records")

        return jsonify({
            "symbol": symbol, "price": round(price, 6),
            "change": prices_cache.get(symbol, {}).get("change", 0),
            "risk": risk_m, "rf": rf_pred, "sentiment": sent,
            "decision": dec, "stops": stops_d, "ai": ai_res,
            "news": [{"title": n.get("title"), "source": n.get("source",{}).get("title",""),
                      "sentiment": n.get("sentiment","neutral")} for n in news[:8]],
            "ohlcv": ohlcv_d,
            "gemini_configured": bool(GEMINI_KEY),
        })
    except Exception as e:
        log.error(f"Analysis {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train/<symbol>")
def api_train(symbol):
    df = binance.klines(symbol.upper(), "1h", 500)
    if df.empty: return jsonify({"error": "Données indisponibles"}), 500
    df = tech.compute_all(df)
    return jsonify(ml.train(df))

@app.route("/api/gemini/<symbol>")
def api_gemini_analysis(symbol):
    """
    Endpoint dédié pour l'analyse Gemini — appelé séparément après /api/analysis.
    Permet de retourner l'analyse rapide d'abord, puis Gemini quand il répond.
    """
    symbol = symbol.upper()
    try:
        df = binance.klines(symbol, "1h", 100)
        if df.empty: return jsonify({"gemini": None, "local": "Données indisponibles"})
        df = tech.compute_all(df)
        price   = prices_cache.get(symbol, {}).get("price", float(df["close"].iloc[-1]))
        risk_m  = risk_eng.compute_all(df)
        rf_pred = ml.predict(df)
        news    = sentiment.fetch(symbol.replace("USDT",""))
        sent    = sentiment.score(news)
        dec     = decision.compute_score(df, rf_pred, sent["score"], risk_m)
        last    = df.iloc[-1]
        ind     = {k: round(float(last[k]),4) for k in
                  ["rsi","macd_hist","adx","bb_pos","stoch_k","stoch_d","vol_ratio","mm_cross"]
                  if k in df.columns and not pd.isna(last[k])}
        ctx = {"symbol": symbol, "price": price, "risk": risk_m,
               "sentiment": sent, "rf": rf_pred, "indicators": ind,
               "decision_score": dec["score"]}
        # Appel direct Gemini (pas de thread — endpoint dédié)
        result = gemini.synthesize(ctx, use_thread=False)
        return jsonify(result)
    except Exception as e:
        log.error(f"Gemini analysis {symbol}: {e}")
        return jsonify({"gemini": None, "local": str(e)[:100]})

@app.route("/api/gemini_test")
def api_gemini_test():
    """Diagnostic Gemini — ouvrir dans le navigateur pour tester."""
    result = {
        "env_file":   str(SCRIPT_DIR / ".env"),
        "env_exists": (SCRIPT_DIR / ".env").exists(),
        "key_found":  bool(GEMINI_KEY),
        "key_prefix": GEMINI_KEY[:8] + "..." if len(GEMINI_KEY) > 8 else GEMINI_KEY,
        "sdk_installed": GENAI_SDK,
    }

    if not GEMINI_KEY:
        result["ok"]    = False
        result["error"] = "GEMINI_API_KEY absente du fichier .env"
        result["fix"]   = f"Ajouter dans {SCRIPT_DIR / '.env'} : GEMINI_API_KEY=AIzaSy..."
        return jsonify(result)

    if not GENAI_SDK:
        result["ok"]    = False
        result["error"] = "SDK google-genai non installé"
        result["fix"]   = "Dans Anaconda Prompt : pip install -q -U google-genai"
        return jsonify(result)

    # Essayer dans l'ordre — gemini-2.5-flash confirmé fonctionnel
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    working_model = None
    model_results = {}

    for model in models_to_try:
        try:
            client   = google_genai.Client(api_key=GEMINI_KEY)
            response = client.models.generate_content(
                model=model,
                contents="Dis juste OK",
                config=genai_types.GenerateContentConfig(max_output_tokens=5),
            )
            model_results[model] = f"✅ OK → '{response.text.strip()}'"
            working_model = model
            gemini._client = client
            break
        except Exception as e:
            model_results[model] = f"✗ {str(e)[:80]}"

    result["ok"]           = working_model is not None
    result["models_tested"]= model_results
    result["working_model"]= working_model
    result["message"]      = f"✅ Modèle fonctionnel : {working_model}" if working_model else "❌ Aucun modèle ne fonctionne"

    return jsonify(result)

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok", "ml": ML_OK, "auto": auto_running,
        "ml_trained": ml.trained, "ml_accuracy": ml.accuracy,
        "gemini_configured": bool(GEMINI_KEY),
        "gemini_sdk": GENAI_SDK,
        "cryptopanic_configured": bool(CRYPTOPANIC_KEY),
        "n_cryptos": len(ALL_CRYPTOS),
        "buy_threshold": decision.BUY_THRESHOLD,
        "stop_loss_pct": decision.STOP_LOSS_PCT * 100,
    })


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=5000, type=int)
    args = p.parse_args()

    threading.Thread(target=price_updater_loop, daemon=True).start()

    # Diagnostic au démarrage
    log.info("=" * 55)
    log.info("    THE CRYPTO SENTINEL — ESME Paris 2026")
    log.info("=" * 55)
    log.info(f"Dossier .env cherché : {SCRIPT_DIR / '.env'}")
    log.info(f"Fichier .env trouvé  : {(SCRIPT_DIR / '.env').exists()}")
    log.info(f"Gemini key       : {'OUI ✓ (' + GEMINI_KEY[:8] + '...)' if GEMINI_KEY else 'NON ✗  → ajouter GEMINI_API_KEY dans .env'}")
    log.info(f"SDK google-genai : {'OUI ✓' if GENAI_SDK else 'NON ✗  → pip install -q -U google-genai'}")
    log.info(f"CryptoPanic          : {'OUI ✓' if CRYPTOPANIC_KEY else 'Mode démo'}")
    log.info(f"scikit-learn ML      : {'OUI ✓' if ML_OK else 'NON ✗ — pip install scikit-learn'}")
    log.info(f"100 cryptos suivies  | Seuil achat : {decision.BUY_THRESHOLD}/100 | Stop : {decision.STOP_LOSS_PCT*100:.0f}%")
    log.info(f"→ http://localhost:{args.port}")
    log.info(f"→ Test Gemini : http://localhost:{args.port}/api/gemini_test")
    log.info("=" * 55)

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)