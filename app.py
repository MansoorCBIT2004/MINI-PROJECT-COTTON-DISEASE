"""
Cotton Leaf Disease Detection System
Multi-page Streamlit Application
Pages: Home | Detection | Results | Metrics | Research | About
"""
 
import streamlit as st
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore")
 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
 
# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --gd:#0d3b1e;--gm:#1a5c32;--gl:#2e8b57;
  --ac:#7fff6e;--ac2:#f5c518;--bg:#060e08;
  --card:#0f1f14;--card2:#142919;--tx:#e8f5e2;
  --mu:#7da88a;--r:14px;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--tx)!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:var(--bg);}::-webkit-scrollbar-thumb{background:var(--gm);border-radius:3px;}
.main .block-container{padding:1.2rem 2rem 2rem 2rem;max-width:1200px;}
/* ── Nav pills ── */
.nav-pill{display:inline-flex;align-items:center;gap:6px;background:rgba(46,139,87,0.12);border:1px solid rgba(46,139,87,0.3);border-radius:22px;padding:5px 16px;font-size:0.82rem;font-weight:600;color:var(--mu);cursor:pointer;margin:3px;}
.nav-pill.active{background:var(--ac);color:#060e08;border-color:var(--ac);}
/* ── Cards ── */
.card{background:var(--card);border:1px solid rgba(46,139,87,0.28);border-radius:var(--r);padding:1.3rem 1.5rem;margin-bottom:1rem;}
.card-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--ac);margin-bottom:0.7rem;display:flex;align-items:center;gap:8px;}
/* ── Hero ── */
.hero{background:linear-gradient(135deg,var(--gd) 0%,#0a2910 50%,#061408 100%);border:1px solid var(--gm);border-radius:20px;padding:2.5rem 3rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::before{content:"";position:absolute;top:-50px;right:-50px;width:220px;height:220px;background:radial-gradient(circle,rgba(46,139,87,0.2) 0%,transparent 70%);border-radius:50%;}
.hero-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:var(--ac);line-height:1.1;margin:0 0 0.4rem;}
.hero-sub{font-size:1rem;color:var(--mu);margin:0;}
.badge{display:inline-block;background:rgba(127,255,110,0.12);border:1px solid var(--ac);color:var(--ac);border-radius:20px;padding:3px 14px;font-size:0.75rem;font-weight:600;margin-bottom:0.8rem;}
/* ── Prediction ── */
.pred-row{display:flex;align-items:center;justify-content:space-between;background:var(--card2);border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.45rem;border:1px solid rgba(46,139,87,0.18);}
.pred-model{font-weight:600;color:var(--mu);font-size:0.82rem;}
.pred-cls{font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--tx);}
.pred-conf{background:rgba(127,255,110,0.12);color:var(--ac);border-radius:20px;padding:2px 11px;font-size:0.8rem;font-weight:600;}
/* ── Final result ── */
.final-box{background:linear-gradient(90deg,rgba(127,255,110,0.14),rgba(127,255,110,0.04));border:2px solid var(--ac);border-radius:var(--r);padding:1.4rem 2rem;text-align:center;margin:1rem 0;}
.final-label{font-size:0.75rem;font-weight:600;letter-spacing:1.5px;color:var(--mu);text-transform:uppercase;margin-bottom:0.3rem;}
.final-cls{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--ac);}
.final-conf{color:var(--mu);font-size:0.88rem;margin-top:0.2rem;}
/* ── Metric card ── */
.mc{background:var(--card2);border:1px solid rgba(46,139,87,0.2);border-radius:10px;padding:0.9rem 1rem;text-align:center;}
.mc-val{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:var(--ac);}
.mc-lbl{font-size:0.72rem;color:var(--mu);font-weight:500;margin-top:2px;}
/* ── Disease ── */
.dis-info{background:var(--card2);border-left:4px solid var(--ac);border-radius:0 var(--r) var(--r) 0;padding:1.1rem 1.3rem;margin-top:0.8rem;}
.pest-badge{display:inline-block;background:rgba(245,197,24,0.12);border:1px solid var(--ac2);color:var(--ac2);border-radius:6px;padding:2px 9px;font-size:0.76rem;font-weight:600;margin:3px 3px 3px 0;}
/* ── Section header ── */
.sh{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:var(--tx);padding-bottom:0.4rem;border-bottom:2px solid var(--gm);margin:1.5rem 0 1rem;}
/* ── Feature card ── */
.feat{background:var(--card);border:1px solid rgba(46,139,87,0.25);border-radius:var(--r);padding:1.2rem;text-align:center;}
.feat-icon{font-size:2rem;margin-bottom:0.5rem;}
.feat-title{font-family:'Syne',sans-serif;font-weight:700;color:var(--ac);font-size:0.95rem;margin-bottom:0.3rem;}
.feat-desc{font-size:0.8rem;color:var(--mu);line-height:1.5;}
/* ── Step card ── */
.step{display:flex;gap:1rem;align-items:flex-start;background:var(--card2);border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.6rem;}
.step-num{background:var(--ac);color:#060e08;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:800;font-size:0.85rem;flex-shrink:0;}
.step-txt{font-size:0.88rem;color:var(--tx);line-height:1.5;}
.step-title{font-weight:600;color:var(--ac);font-size:0.9rem;margin-bottom:2px;}
/* sidebar */
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid rgba(46,139,87,0.25);}
[data-testid="stFileUploader"]{background:var(--card2)!important;border:2px dashed var(--gm)!important;border-radius:var(--r)!important;}
#MainMenu,footer{visibility:hidden;}
/* page indicator */
.page-indicator{display:flex;gap:6px;justify-content:center;margin-bottom:1.2rem;}
.pi-dot{width:8px;height:8px;border-radius:50%;background:rgba(46,139,87,0.3);}
.pi-dot.active{background:var(--ac);width:24px;border-radius:4px;}
</style>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
CLASS_DISPLAY = ["Bacterial Blight", "Curl Virus", "Fussarium Wilt", "Healthy"]
 
MODEL_PATHS = {
    "CNN (MobileNetV2)": "models/cnn_model.h5",
    "CNN-LSTM":          "models/cnn_lstm_model.h5",
    "ResNet50":          "models/resnet_model.h5",
}
 
# ⚠️ Replace with your actual classification_report values
PROPOSED_METRICS = {
    "CNN (MobileNetV2)": {"Accuracy":0.8833,"Precision":0.899,"Recall":0.885,"F1-Score":0.881},
    "CNN-LSTM":          {"Accuracy":0.6530,"Precision":0.653,"Recall":0.653,"F1-Score":0.641},
    "ResNet50":          {"Accuracy":0.9529,"Precision":0.953,"Recall":0.953,"F1-Score":0.952},
}
 
EXISTING_METRICS = {
    "RFCottonNet": {"Accuracy":0.9210,"Precision":0.9190,"Recall":0.9150,"F1-Score":0.9170},
    "FSeMNet":     {"Accuracy":0.9380,"Precision":0.9340,"Recall":0.9320,"F1-Score":0.9330},
    "YOLO":        {"Accuracy":0.9100,"Precision":0.9060,"Recall":0.9040,"F1-Score":0.9050},
}
 
DISEASE_INFO = {
    "Bacterial Blight":{"icon":"🦠","severity":"High",
        "description":"Caused by Xanthomonas citri pv. malvacearum. Produces angular, water-soaked lesions on leaves that turn brown and necrotic. Spreads rapidly in warm, humid conditions.",
        "pesticides":["Copper Oxychloride (0.3%)","Streptomycin Sulphate (100 ppm)","Bacterinol-100"],
        "cultural":"Remove infected debris. Avoid overhead irrigation. Use disease-free seeds."},
    "Curl Virus":{"icon":"🌀","severity":"Very High",
        "description":"Cotton Leaf Curl Virus (CLCuV) transmitted by whitefly Bemisia tabaci. Shows upward/downward curling of leaves, dark green veins, and vein-enation on underside.",
        "pesticides":["Imidacloprid 70 WS","Thiamethoxam 25 WG","Acetamiprid 20 SP"],
        "cultural":"Control whitefly populations. Use resistant varieties. Remove alternate hosts."},
    "Fussarium Wilt":{"icon":"🍂","severity":"High",
        "description":"Fusarium oxysporum f. sp. vasinfectum causes yellowing from leaf margins inward. Vascular discoloration visible in stem cross-sections. Soil-borne and persistent.",
        "pesticides":["Carbendazim 50 WP (soil drench)","Trichoderma viride","Thiophanate-methyl 70 WP"],
        "cultural":"Use certified disease-free seeds. Crop rotation. Soil solarization before planting."},
    "Healthy":{"icon":"✅","severity":"None",
        "description":"No visible disease signs. The cotton leaf is healthy. Continue regular monitoring and preventive agronomic practices.",
        "pesticides":[],"cultural":"Maintain irrigation, balanced fertilization, and pest scouting."},
}
 
# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_all_models():
    from tensorflow.keras import layers
    from tensorflow.keras.models import load_model as klm
 
    class PatchedInputLayer(layers.InputLayer):
        def __init__(self,*a,**kw):
            bs=kw.pop('batch_shape',None)
            kw.pop('optional',None);kw.pop('sparse',None);kw.pop('ragged',None)
            if bs is not None and 'shape' not in kw: kw['shape']=tuple(bs[1:])
            super().__init__(*a,**kw)
 
    class PatchedDense(layers.Dense):
        def __init__(self,*a,**kw): kw.pop('quantization_config',None); super().__init__(*a,**kw)
 
    class PatchedConv2D(layers.Conv2D):
        def __init__(self,*a,**kw): kw.pop('quantization_config',None); super().__init__(*a,**kw)
 
    class PatchedDWConv2D(layers.DepthwiseConv2D):
        def __init__(self,*a,**kw): kw.pop('quantization_config',None); super().__init__(*a,**kw)
 
    class PatchedBN(layers.BatchNormalization):
        def __init__(self,*a,**kw): kw.pop('quantization_config',None); super().__init__(*a,**kw)
 
    class PatchedLSTM(layers.LSTM):
        def __init__(self,*a,**kw): kw.pop('quantization_config',None); super().__init__(*a,**kw)
 
    co={'InputLayer':PatchedInputLayer,'Dense':PatchedDense,'Conv2D':PatchedConv2D,
        'DepthwiseConv2D':PatchedDWConv2D,'BatchNormalization':PatchedBN,'LSTM':PatchedLSTM}
 
    loaded,errors={},[]
    for name,path in MODEL_PATHS.items():
        if os.path.exists(path):
            try: loaded[name]=klm(path,compile=False,custom_objects=co)
            except Exception as e: errors.append(f"❌ **{name}**: {e}")
        else: errors.append(f"⚠️ Not found: `{path}`")
    return loaded,errors
 
# ══════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def crop_leaf(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([20,30,30]),np.array([95,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
        c=bgr[y:y+h,x:x+w]
        if c.size>0: return c
    return bgr
 
def enhance_contrast(bgr):
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    cl=cv2.createCLAHE(3.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2BGR)
 
def preprocess_image(pil_img):
    rgb=np.array(pil_img.convert("RGB"))
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    bgr=crop_leaf(bgr); bgr=enhance_contrast(bgr)
    inp={}
    # CNN MobileNetV2: 224x224
    c=cv2.resize(bgr,(224,224)).astype(np.float32)
    inp["CNN (MobileNetV2)"]=np.expand_dims(mobilenet_preprocess(c),0)
    # CNN-LSTM: 128x128 /255 reshape
    l=cv2.resize(bgr,(128,128)).astype(np.float32)/255.0
    inp["CNN-LSTM"]=l.reshape(1,128,384)
    # ResNet50: 224x224
    r=cv2.resize(bgr,(224,224)).astype(np.float32)
    inp["ResNet50"]=np.expand_dims(resnet_preprocess(r),0)
    return inp
 
def run_predictions(models_dict,inputs):
    results={}
    for name,model in models_dict.items():
        if name not in inputs: continue
        try:
            p=model.predict(inputs[name],verbose=0)[0]
            if p.sum()>0: p=p/p.sum()
            idx=int(np.argmax(p))
            results[name]={"probs":p,"pred_idx":idx,"pred_class":CLASS_DISPLAY[idx],"confidence":float(p[idx])}
        except Exception as e:
            st.warning(f"Prediction error for {name}: {e}")
    return results
 
def ensemble_vote(results):
    if not results: return None,0.0,None
    vc,all_p={},[]
    for r in results.values():
        vc[r["pred_class"]]=vc.get(r["pred_class"],0)+1
        all_p.append(r["probs"])
    mv=max(vc.values())
    tops=[c for c,v in vc.items() if v==mv]
    avg=np.mean(all_p,axis=0)
    final=tops[0] if len(tops)==1 else CLASS_DISPLAY[max([CLASS_DISPLAY.index(c) for c in tops],key=lambda i:avg[i])]
    return final,float(avg[CLASS_DISPLAY.index(final)]),avg
 
# ══════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════
def prob_chart(avg,pred):
    fig,ax=plt.subplots(figsize=(6.5,3))
    fig.patch.set_facecolor("#0f1f14"); ax.set_facecolor("#0f1f14")
    colors=["#7fff6e" if c==pred else "#2e8b57" for c in CLASS_DISPLAY]
    bars=ax.barh(CLASS_DISPLAY,avg*100,color=colors,edgecolor="none",height=0.5)
    for bar,p in zip(bars,avg):
        ax.text(min(p*100+1,97),bar.get_y()+bar.get_height()/2,f"{p*100:.1f}%",
                va="center",ha="left",color="#e8f5e2",fontsize=9,fontweight="600")
    ax.set_xlim(0,105); ax.set_xlabel("Confidence (%)",color="#7da88a",fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="x",colors="#7da88a"); ax.tick_params(axis="y",colors="#e8f5e2",labelsize=9)
    ax.grid(axis="x",color="#1a5c32",linestyle="--",alpha=0.4)
    fig.tight_layout(pad=1.0); return fig
 
def acc_chart(all_m,metric="Accuracy"):
    fig,ax=plt.subplots(figsize=(8.5,3.8))
    fig.patch.set_facecolor("#0f1f14"); ax.set_facecolor("#0f1f14")
    names=list(PROPOSED_METRICS)+list(EXISTING_METRICS)
    vals=[all_m[n][metric]*100 for n in names]
    colors=["#7fff6e"]*len(PROPOSED_METRICS)+["#f5a623"]*len(EXISTING_METRICS)
    bars=ax.bar(names,vals,color=colors,width=0.5,edgecolor="none")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{v:.1f}%",
                ha="center",va="bottom",color="#e8f5e2",fontsize=8,fontweight="600")
    ax.set_ylim(60,100); ax.set_ylabel(f"{metric} (%)",color="#7da88a",fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y",color="#1a5c32",linestyle="--",alpha=0.4)
    ax.tick_params(axis="x",colors="#e8f5e2",labelsize=8); ax.tick_params(axis="y",colors="#7da88a")
    ax.legend(handles=[mpatches.Patch(color="#7fff6e",label="Proposed"),mpatches.Patch(color="#f5a623",label="Literature")],
              facecolor="#0f1f14",edgecolor="#2e8b57",labelcolor="#e8f5e2",fontsize=8)
    fig.tight_layout(pad=1.0); return fig
 
def pr_chart(all_m):
    fig,ax=plt.subplots(figsize=(8.5,3.8))
    fig.patch.set_facecolor("#0f1f14"); ax.set_facecolor("#0f1f14")
    names=list(PROPOSED_METRICS)+list(EXISTING_METRICS)
    x=np.arange(len(names)); w=0.33
    b1=ax.bar(x-w/2,[all_m[n]["Precision"]*100 for n in names],w,label="Precision",color="#7fff6e",alpha=0.9)
    b2=ax.bar(x+w/2,[all_m[n]["Recall"]*100    for n in names],w,label="Recall",   color="#4ec9ff",alpha=0.9)
    for bars in [b1,b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                    f"{bar.get_height():.1f}",ha="center",va="bottom",color="#e8f5e2",fontsize=7,fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha="right",color="#e8f5e2",fontsize=8)
    ax.set_ylim(60,100); ax.set_ylabel("Score (%)",color="#7da88a",fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y",color="#1a5c32",linestyle="--",alpha=0.4); ax.tick_params(axis="y",colors="#7da88a")
    ax.legend(facecolor="#0f1f14",edgecolor="#2e8b57",labelcolor="#e8f5e2",fontsize=8)
    fig.tight_layout(pad=1.0); return fig
 
# ══════════════════════════════════════════════════════════════════
#  SIDEBAR NAV
# ══════════════════════════════════════════════════════════════════
PAGES = ["🏠 Home", "🔬 Detection", "📊 Results", "📈 Metrics", "🔍 Research", "ℹ️ About"]
 
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#7fff6e;">🌿 CottonAI</div>
      <div style="font-size:0.72rem;color:#7da88a;margin-top:2px;">Disease Detection System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
 
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
 
    for p in PAGES:
        active = st.session_state.page == p
        if st.button(p, key=f"nav_{p}",
                     type="primary" if active else "secondary",
                     use_container_width=True):
            st.session_state.page = p
            st.rerun()
 
    st.markdown("---")
    # Model loading status in sidebar
    with st.spinner("Loading models..."):
        models, load_errors = load_all_models()
 
    if models:
        st.success(f"✅ {len(models)} models ready")
    for e in load_errors:
        st.warning(e)
 
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#7da88a;line-height:1.8;">
    <b style="color:#7fff6e;">Classes:</b><br>
    🦠 Bacterial Blight<br>
    🌀 Curl Virus<br>
    🍂 Fussarium Wilt<br>
    ✅ Healthy
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🌱 Agriculture AI · Deep Learning")
 
page = st.session_state.page
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <div class="badge">🌿 Deep Learning · Agriculture AI</div>
      <h1 class="hero-title">Cotton Leaf Disease<br>Detection System</h1>
      <p class="hero-sub">AI-powered disease detection using CNN, CNN-LSTM & ResNet50 ensemble models</p>
    </div>
    """, unsafe_allow_html=True)
 
    # Feature cards
    st.markdown('<div class="sh">✨ Key Features</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    feats = [
        ("🧠","3 AI Models","CNN, CNN-LSTM & ResNet50 work together for accurate detection"),
        ("⚡","Instant Results","Upload a photo and get disease diagnosis in seconds"),
        ("💊","Treatment Guide","Get pesticide & cultural practice recommendations"),
        ("📊","Research Backed","Compared with published research models"),
    ]
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],feats):
        with col:
            st.markdown(f"""
            <div class="feat">
              <div class="feat-icon">{icon}</div>
              <div class="feat-title">{title}</div>
              <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
 
    # How it works
    st.markdown('<div class="sh">⚙️ How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("Upload Image","Go to the Detection page and upload or scan a cotton leaf photo using camera or gallery."),
        ("Preprocessing","Image is cropped using HSV masking, contrast-enhanced with CLAHE, then resized for each model."),
        ("AI Inference","All 3 models (CNN MobileNetV2, CNN-LSTM, ResNet50) run predictions simultaneously."),
        ("Ensemble Voting","Majority voting with tie-break by average probability gives the final prediction."),
        ("Results & Treatment","View disease info, confidence scores, and recommended pesticides on the Results page."),
    ]
    c_left, c_right = st.columns(2)
    for i,(t,d) in enumerate(steps):
        col = c_left if i%2==0 else c_right
        with col:
            st.markdown(f"""
            <div class="step">
              <div class="step-num">{i+1}</div>
              <div class="step-txt"><div class="step-title">{t}</div>{d}</div>
            </div>""", unsafe_allow_html=True)
 
    # Disease classes
    st.markdown('<div class="sh">🌿 Detectable Diseases</div>', unsafe_allow_html=True)
    dc1,dc2,dc3,dc4 = st.columns(4)
    diseases = [
        ("🦠","Bacterial Blight","High","Caused by Xanthomonas bacteria"),
        ("🌀","Curl Virus","Very High","Spread by whitefly Bemisia tabaci"),
        ("🍂","Fussarium Wilt","High","Soil-borne fungal infection"),
        ("✅","Healthy","None","No disease detected"),
    ]
    sev_colors = {"High":"#f5a623","Very High":"#ff4e4e","None":"#7fff6e"}
    for col,(icon,name,sev,desc) in zip([dc1,dc2,dc3,dc4],diseases):
        with col:
            sc = sev_colors[sev]
            st.markdown(f"""
            <div class="feat">
              <div class="feat-icon">{icon}</div>
              <div class="feat-title">{name}</div>
              <div style="display:inline-block;background:rgba(0,0,0,0.3);border:1px solid {sc};color:{sc};border-radius:6px;padding:1px 8px;font-size:0.7rem;margin-bottom:6px;">Severity: {sev}</div>
              <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("""
    <div style="text-align:center;margin-top:2rem;">
      <div style="font-size:0.85rem;color:#7da88a;">👈 Use the sidebar to navigate between pages</div>
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — DETECTION
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Detection":
    st.markdown('<div class="sh">🔬 Cotton Leaf Detection</div>', unsafe_allow_html=True)
 
    if not models:
        st.error("❌ No models loaded. Check that `models/` folder has the `.h5` files.")
        st.stop()
 
    # Camera + Gallery tabs
    tab_cam, tab_gallery = st.tabs(["📷 Scan / Camera", "🖼️ Upload from Gallery"])
    uploaded_file = None
 
    with tab_cam:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📱 Point your camera at the cotton leaf and capture**")
        cam_img = st.camera_input("Capture cotton leaf")
        if cam_img:
            uploaded_file = cam_img
        st.markdown("</div>", unsafe_allow_html=True)
 
    with tab_gallery:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🖼️ Select an existing photo from your device**")
        gallery_img = st.file_uploader("Choose JPG or PNG image", type=["jpg","jpeg","png"])
        if gallery_img:
            uploaded_file = gallery_img
        st.markdown("</div>", unsafe_allow_html=True)
 
    if uploaded_file is not None:
        # Validate
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.verify()
            pil_image = Image.open(uploaded_file)
        except Exception:
            st.error("❌ Invalid image file. Please upload a valid JPG or PNG.")
            st.stop()
 
        col_img, col_run = st.columns([1, 1.2], gap="large")
 
        with col_img:
            st.markdown('<div class="card"><div class="card-title">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
            st.markdown(f'<div style="color:var(--mu);font-size:0.78rem;margin-top:4px;">Size: {pil_image.size[0]}×{pil_image.size[1]} px</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
 
        with col_run:
            st.markdown('<div class="card"><div class="card-title">🚀 Run Detection</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.85rem;color:var(--mu);line-height:1.7;margin-bottom:1rem;">
            ✅ Image loaded successfully<br>
            ✅ {len(models)} models ready<br>
            ✅ Preprocessing pipeline ready<br>
            </div>""", unsafe_allow_html=True)
 
            if st.button("🔬 Analyze Leaf Now", type="primary", use_container_width=True):
                with st.spinner("🔬 Preprocessing & running inference..."):
                    try:
                        inp = preprocess_image(pil_image)
                        results = run_predictions(models, inp)
                        final_class, final_conf, avg_probs = ensemble_vote(results)
                        # Save to session state
                        st.session_state.results = results
                        st.session_state.final_class = final_class
                        st.session_state.final_conf = final_conf
                        st.session_state.avg_probs = avg_probs
                        st.session_state.pil_image = pil_image
                        st.session_state.analysis_done = True
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        st.stop()
 
                st.success("✅ Analysis complete! Go to **Results** page to see predictions.")
                st.balloons()
 
            st.markdown("</div>", unsafe_allow_html=True)
 
        # Quick preview if already done
        if st.session_state.get("analysis_done"):
            fc = st.session_state.final_class
            info = DISEASE_INFO.get(fc,{})
            st.markdown(f"""
            <div class="final-box" style="margin-top:1rem;">
              <div class="final-label">Last Analysis Result</div>
              <div class="final-cls">{info.get('icon','')} {fc}</div>
              <div class="final-conf">Confidence: {st.session_state.final_conf*100:.1f}% · Click Results page for details</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">📷</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--ac);margin-bottom:0.5rem;">No Image Selected</div>
          <div style="font-size:0.85rem;color:var(--mu);">Use the Camera tab to scan or Gallery tab to upload a cotton leaf image above</div>
        </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — RESULTS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Results":
    st.markdown('<div class="sh">📊 Detection Results</div>', unsafe_allow_html=True)
 
    if not st.session_state.get("analysis_done"):
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--ac);margin-bottom:0.5rem;">No Analysis Yet</div>
          <div style="font-size:0.85rem;color:var(--mu);">Please go to the Detection page first and analyze a cotton leaf image.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Go to Detection", type="primary"):
            st.session_state.page = "🔬 Detection"
            st.rerun()
        st.stop()
 
    results    = st.session_state.results
    final_class= st.session_state.final_class
    final_conf = st.session_state.final_conf
    avg_probs  = st.session_state.avg_probs
    pil_image  = st.session_state.pil_image
    info       = DISEASE_INFO.get(final_class,{})
 
    # Top section
    col_img, col_pred = st.columns([1, 1.4], gap="large")
 
    with col_img:
        st.markdown('<div class="card"><div class="card-title">🖼️ Analyzed Image</div>', unsafe_allow_html=True)
        st.image(pil_image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col_pred:
        st.markdown('<div class="card"><div class="card-title">🤖 Model Predictions</div>', unsafe_allow_html=True)
        for mname, res in results.items():
            st.markdown(f"""
            <div class="pred-row">
              <span class="pred-model">{mname}</span>
              <span class="pred-cls">{res['pred_class']}</span>
              <span class="pred-conf">{res['confidence']*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
 
        st.markdown(f"""
        <div class="final-box">
          <div class="final-label">🏆 Ensemble Final Prediction</div>
          <div class="final-cls">{info.get('icon','')} {final_class}</div>
          <div class="final-conf">Ensemble Confidence: {final_conf*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Confidence chart
    st.markdown('<div class="sh">📊 Class Confidence Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Average probability across all three models")
    fig = prob_chart(avg_probs, final_class)
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # Disease info
    st.markdown('<div class="sh">🩺 Disease Information & Treatment</div>', unsafe_allow_html=True)
    if info:
        sev_color={"None":"#7fff6e","High":"#f5a623","Very High":"#ff4e4e"}.get(info["severity"],"#f5a623")
        st.markdown(f"""
        <div class="card">
          <div class="card-title">{info['icon']} {final_class}
            <span style="margin-left:auto;background:rgba(0,0,0,0.3);border:1px solid {sev_color};color:{sev_color};border-radius:6px;padding:2px 10px;font-size:0.72rem;">Severity: {info['severity']}</span>
          </div>
          <div class="dis-info">
            <p style="color:var(--tx);margin:0 0 0.7rem;line-height:1.65;">{info['description']}</p>
            <p style="color:var(--mu);font-size:0.78rem;font-weight:600;margin-bottom:0.25rem;">🌾 Cultural Practice:</p>
            <p style="color:var(--tx);font-size:0.86rem;margin:0 0 0.7rem;">{info['cultural']}</p>
        """, unsafe_allow_html=True)
        if info["pesticides"]:
            st.markdown('<p style="color:var(--mu);font-size:0.78rem;font-weight:600;">💊 Recommended Pesticides:</p>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="pest-badge">{p}</span>' for p in info["pesticides"]]), unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(127,255,110,0.12);border:1px solid var(--ac);color:var(--ac);border-radius:8px;padding:0.5rem 1rem;font-weight:600;display:inline-block;">✅ Plant is healthy — no treatment needed!</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
 
    # Detailed table
    with st.expander("🔍 Detailed Probability Breakdown per Class"):
        rows=[]
        for i,cls in enumerate(CLASS_DISPLAY):
            row={"Class":cls}
            for mn,res in results.items(): row[mn]=f"{res['probs'][i]*100:.2f}%"
            row["Avg (Ensemble)"]=f"{avg_probs[i]*100:.2f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Class"),use_container_width=True)
 
    # Re-analyze button
    if st.button("🔄 Analyze Another Image", use_container_width=True):
        st.session_state.analysis_done = False
        st.session_state.page = "🔬 Detection"
        st.rerun()
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — METRICS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Metrics":
    st.markdown('<div class="sh">📈 Model Evaluation Metrics</div>', unsafe_allow_html=True)
    st.warning("⚠️ Update `PROPOSED_METRICS` in app.py with your actual `classification_report()` values from training notebooks.")
 
    for model_name, metrics in PROPOSED_METRICS.items():
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🤖 {model_name}</div>
        """, unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,(metric,val) in zip([c1,c2,c3,c4],metrics.items()):
            with col:
                st.markdown(f"""
                <div class="mc">
                  <div class="mc-val">{val*100:.1f}%</div>
                  <div class="mc-lbl">{metric}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Per-model accuracy bar
    st.markdown('<div class="sh">📊 Accuracy Comparison — Proposed Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(7,3))
    fig.patch.set_facecolor("#0f1f14"); ax.set_facecolor("#0f1f14")
    names=list(PROPOSED_METRICS.keys())
    accs=[PROPOSED_METRICS[n]["Accuracy"]*100 for n in names]
    bars=ax.bar(names,accs,color=["#7fff6e","#4ec9ff","#f5a623"],width=0.45,edgecolor="none")
    for bar,v in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{v:.1f}%",
                ha="center",va="bottom",color="#e8f5e2",fontsize=10,fontweight="600")
    ax.set_ylim(50,100); ax.set_ylabel("Accuracy (%)",color="#7da88a",fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y",color="#1a5c32",linestyle="--",alpha=0.4)
    ax.tick_params(axis="x",colors="#e8f5e2"); ax.tick_params(axis="y",colors="#7da88a")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # Summary table
    st.markdown('<div class="sh">📋 Full Metrics Table</div>', unsafe_allow_html=True)
    rows=[]
    for name,m in PROPOSED_METRICS.items():
        rows.append({"Model":name,"Accuracy":f"{m['Accuracy']*100:.1f}%",
                     "Precision":f"{m['Precision']*100:.1f}%",
                     "Recall":f"{m['Recall']*100:.1f}%",
                     "F1-Score":f"{m['F1-Score']*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 5 — RESEARCH
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Research":
    st.markdown('<div class="sh">🔍 Research Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="padding:0.8rem 1.2rem;margin-bottom:1rem;">
      <span style="font-size:0.82rem;color:var(--mu);">
        ✅ <b style="color:var(--ac);">Proposed Models</b> — trained locally on cotton dataset &nbsp;|&nbsp;
        📄 <b style="color:#f5a623;">Existing Models</b> — metrics from published research papers
      </span>
    </div>
    """, unsafe_allow_html=True)
 
    all_m = {**PROPOSED_METRICS, **EXISTING_METRICS}
 
    tab1,tab2,tab3,tab4 = st.tabs(["📋 Table","📊 Accuracy","📉 Precision & Recall","📈 F1-Score"])
 
    with tab1:
        rows=[]
        for n,m in PROPOSED_METRICS.items():
            rows.append({"Model":f"✅ {n}","Type":"Proposed","Accuracy":f"{m['Accuracy']*100:.1f}%",
                         "Precision":f"{m['Precision']*100:.1f}%","Recall":f"{m['Recall']*100:.1f}%","F1-Score":f"{m['F1-Score']*100:.1f}%"})
        for n,m in EXISTING_METRICS.items():
            rows.append({"Model":f"📄 {n}","Type":"Literature","Accuracy":f"{m['Accuracy']*100:.1f}%",
                         "Precision":f"{m['Precision']*100:.1f}%","Recall":f"{m['Recall']*100:.1f}%","F1-Score":f"{m['F1-Score']*100:.1f}%"})
        st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
 
    with tab2:
        fig=acc_chart(all_m,"Accuracy")
        st.pyplot(fig,use_container_width=True); plt.close(fig)
 
    with tab3:
        fig=pr_chart(all_m)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
 
    with tab4:
        fig=acc_chart(all_m,"F1-Score")
        st.pyplot(fig,use_container_width=True); plt.close(fig)
 
    # Conclusion
    st.markdown('<div class="sh">🏁 Conclusion</div>', unsafe_allow_html=True)
    bp=max(PROPOSED_METRICS,key=lambda k:PROPOSED_METRICS[k]["Accuracy"])
    bp_acc=PROPOSED_METRICS[bp]["Accuracy"]*100
    be=max(EXISTING_METRICS,key=lambda k:EXISTING_METRICS[k]["Accuracy"])
    be_acc=EXISTING_METRICS[be]["Accuracy"]*100
    imp=bp_acc-be_acc
 
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="mc"><div class="mc-val" style="font-size:1rem;">🥇 {bp}</div><div class="mc-lbl">Best Proposed Model</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="mc"><div class="mc-val">{bp_acc:.1f}%</div><div class="mc-lbl">Best Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="mc"><div class="mc-val" style="color:var(--ac2);">{imp:+.1f}%</div><div class="mc-lbl">vs Best Literature ({be})</div></div>', unsafe_allow_html=True)
 
    st.markdown(f"""
    <div class="card" style="margin-top:1rem;line-height:1.7;font-size:0.9rem;color:var(--tx);">
      The ensemble of <b style="color:var(--ac);">CNN (MobileNetV2)</b>, <b style="color:var(--ac);">CNN-LSTM</b>,
      and <b style="color:var(--ac);">ResNet50</b> with majority voting provides robust cotton disease classification.
      <b>{bp}</b> achieves the highest accuracy of <b>{bp_acc:.1f}%</b>, outperforming the best literature model
      (<b>{be}</b> at {be_acc:.1f}%) by <b style="color:var(--ac2);">{imp:+.1f}%</b>.
      Transfer learning with ImageNet pre-trained weights significantly boosts detection performance for plant disease identification.
    </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════
#  PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="sh">ℹ️ About This System</div>', unsafe_allow_html=True)
 
    col1,col2 = st.columns([1.2,1],gap="large")
 
    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">📌 Project Overview</div>
          <div style="font-size:0.88rem;color:var(--tx);line-height:1.75;">
            This system detects cotton leaf diseases from uploaded images using three trained deep learning models.
            The ensemble approach combines predictions from CNN (MobileNetV2), CNN-LSTM, and ResNet50 to provide
            accurate and reliable disease classification.<br><br>
            The dataset used is the <b style="color:var(--ac);">Cotton Leaf Disease Dataset</b> from Kaggle,
            containing images of 4 classes: Bacterial Blight, Curl Virus, Fussarium Wilt, and Healthy leaves.
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        <div class="card" style="margin-top:0;">
          <div class="card-title">🔧 Preprocessing Pipeline</div>
          <div style="font-size:0.85rem;color:var(--tx);line-height:1.75;">
            <b style="color:var(--ac);">Step 1 — HSV Leaf Cropping:</b><br>
            HSV color masking (H:20-95, S:30-255, V:30-255) detects green leaf regions.
            Morphological closing removes noise. Largest contour is cropped.<br><br>
            <b style="color:var(--ac);">Step 2 — CLAHE Enhancement:</b><br>
            Contrast Limited Adaptive Histogram Equalization (clipLimit=3.0, tile 8×8)
            applied on LAB L-channel for consistent contrast across images.<br><br>
            <b style="color:var(--ac);">Step 3 — Model-specific Normalization:</b><br>
            CNN & ResNet50 use framework preprocess_input. CNN-LSTM divides by 255.
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">🤖 Model Architectures</div>
        """, unsafe_allow_html=True)
        archs = [
            ("🧠","CNN (MobileNetV2)","224×224","MobileNetV2 base (ImageNet) + GlobalAvgPool + Dense(128) + Dropout(0.5) + Softmax(4)","mobilenet_v2 preprocess_input"),
            ("🔁","CNN-LSTM","128×128","2× LSTM(128,64) + Dense(64) + Dropout + Softmax(4)","Normalize /255, reshape (1,128,384)"),
            ("🏗️","ResNet50","224×224","ResNet50 base (ImageNet) + GlobalAvgPool + BatchNorm + Dense(256) + Dropout(0.5) + Softmax(4)","resnet50 preprocess_input"),
        ]
        for icon,name,size,arch,norm in archs:
            st.markdown(f"""
            <div style="background:var(--card2);border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;border:1px solid rgba(46,139,87,0.18);">
              <div style="font-weight:600;color:var(--ac);font-size:0.88rem;margin-bottom:4px;">{icon} {name} ({size})</div>
              <div style="font-size:0.78rem;color:var(--tx);line-height:1.5;">{arch}</div>
              <div style="font-size:0.73rem;color:var(--mu);margin-top:3px;">Preprocessing: {norm}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("""
        <div class="card">
          <div class="card-title">📚 Dataset</div>
          <div style="font-size:0.85rem;color:var(--tx);line-height:1.75;">
            <b style="color:var(--ac);">Dataset:</b> Cotton Leaf Disease Dataset (Kaggle)<br>
            <b style="color:var(--ac);">Source:</b> seroshkarim/cotton-leaf-disease-dataset<br>
            <b style="color:var(--ac);">Classes:</b> 4 (Bacterial Blight, Curl Virus, Fussarium Wilt, Healthy)<br>
            <b style="color:var(--ac);">Split:</b> 80% Train / 20% Test (stratified)<br>
            <b style="color:var(--ac);">Augmentation:</b> Rotation, Zoom, Flip, Shift
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("""
    <hr style="border-color:rgba(46,139,87,0.25);margin-top:2rem;"/>
    <div style="text-align:center;color:var(--mu);font-size:0.78rem;padding:0.8rem 0;">
      🌿 Cotton Leaf Disease Detection System · CNN (MobileNetV2) · CNN-LSTM · ResNet50 · Built with Streamlit & TensorFlow
    </div>
    """, unsafe_allow_html=True)
