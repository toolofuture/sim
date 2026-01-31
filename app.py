import streamlit as st
import time
import os
import sys
from PIL import Image
import plotly.graph_objects as go
import math

# Add root to path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.met_api import MetArtFetcher, FakeGenerator
from core.pipeline import VerificationPipeline
from core.report_generator import create_pdf_report
from core.evolution import EvolutionManager

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AURA | Forensic Art Verification",
    page_icon="🎨",
    layout="wide"
)

# --- LOAD CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def init_system():
    return VerificationPipeline(), MetArtFetcher(), EvolutionManager()

pipeline, fetcher, evolution = init_system()

# --- SESSION STATE MANAGEMENT ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'ref_image' not in st.session_state:
    st.session_state.ref_image = None
if 'suspect_image' not in st.session_state:
    st.session_state.suspect_image = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'suspect_path' not in st.session_state:
    st.session_state.suspect_path = "data/temp_suspect.jpg"

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def reset_system():
    st.session_state.step = 1
    st.session_state.ref_image = None
    st.session_state.suspect_image = None
    st.session_state.results = None
    st.rerun()

# --- SIDEBAR: STATUS & LOGO ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/The_Metropolitan_Museum_of_Art_Logo.svg/1200px-The_Metropolitan_Museum_of_Art_Logo.svg.png", width=180)
    st.markdown("<h2 style='text-align: center; color: #00d4ff;'>AURA V2.0</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.info(f"📍 Current Phase: Step {st.session_state.step}")
    
    with st.expander("System Internals"):
        st.write("Quantum Core: **LOCKED**")
        st.write("Evolution Memory: **ACTIVE**")
        st.Progress(st.session_state.step * 20)
    
    if st.button("🔄 FULL RESET", use_container_width=True):
        reset_system()

# --- MAIN UI ---
st.title("🎨 AURA: SEQUENTIAL ART FORENSICS")
st.markdown("<p style='color: #8b949e;'>Artificial intelligence & Quantum Entropy Verification System</p>", unsafe_allow_html=True)

# Progress Bar
st.progress(st.session_state.step / 5)
cols = st.columns(5)
step_names = ["1. Reference", "2. Target", "3. Analysis", "4. Feedback", "5. Certificate"]
for i, name in enumerate(step_names):
    if st.session_state.step == i+1:
        cols[i].markdown(f"**{name}**")
    else:
        cols[i].markdown(f"<span style='color: #444;'>{name}</span>", unsafe_allow_html=True)

st.markdown("---")

# --- STEP 1: REFERENCE ARTWORK ---
if st.session_state.step == 1:
    st.markdown("<div class='step-header'>STEP 1: ESTABLISH AUTHENTIC REFERENCE</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Manual Upload")
        ref_file = st.file_uploader("Upload known authentic image", type=["jpg", "png", "jpeg"])
        if ref_file:
            st.session_state.ref_image = Image.open(ref_file).convert('RGB')
            st.image(st.session_state.ref_image, caption="Uploaded Reference")

    with col_b:
        st.subheader("Met Database Fetch")
        if st.button("Fetch Random Masterpiece from Met API"):
            with st.spinner("Accessing API..."):
                paths = fetcher.fetch_authentic_samples(count=1)
                if paths:
                    st.session_state.ref_image = Image.open(paths[0]).convert('RGB')
                    st.success("Fetched from Met Collection!")
                else:
                    st.error("Failed to connect to Met API.")
        
        if st.session_state.ref_image:
             st.image(st.session_state.ref_image, caption="Database Reference", use_container_width=True)

    if st.session_state.ref_image:
        if st.button("CONFIRM REFERENCE & PROCEED →"):
            next_step()
            st.rerun()

# --- STEP 2: SUSPECT TARGET ---
elif st.session_state.step == 2:
    st.markdown("<div class='step-header'>STEP 2: UPLOAD SUSPECT ARTWORK</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Authentic Sample (Reference)")
        st.image(st.session_state.ref_image, use_container_width=True)
        
    with col_b:
        st.subheader("Analysis Target")
        suspect_file = st.file_uploader("Upload the piece to be analyzed", type=["jpg", "png", "jpeg"])
        if suspect_file:
            st.session_state.suspect_image = Image.open(suspect_file).convert('RGB')
            st.session_state.suspect_image.save(st.session_state.suspect_path)
            st.image(st.session_state.suspect_image, caption="Ready for Analysis")
        
        st.markdown("---")
        if st.button("🧬 GENERATE SYNTHETIC FAKE (DEMO MODE)"):
             # For when user has no fake image to test
             temp_ref_path = "data/temp_ref.jpg"
             st.session_state.ref_image.save(temp_ref_path)
             st.session_state.suspect_image = FakeGenerator.generate_fake(temp_ref_path)
             st.session_state.suspect_image.save(st.session_state.suspect_path)
             st.success("Synthetic forgery created for testing purposes.")
             st.image(st.session_state.suspect_image, caption="AI-Generated Forgery")

    if st.session_state.suspect_image:
        col1, col2 = st.columns(2)
        if col1.button("← GO BACK"):
            prev_step()
            st.rerun()
        if col2.button("INITIATE SCAN PROTOCOL →"):
            next_step()
            st.rerun()

# --- STEP 3: ANALYSIS & RESULTS ---
elif st.session_state.step == 3:
    st.markdown("<div class='step-header'>STEP 3: QUANTUM & NEURAL FORENSICS</div>", unsafe_allow_html=True)
    
    if not st.session_state.results:
        with st.status("Performing Multimodal Verification...", expanded=True) as status:
            st.write("Extracting Neural Feature Maps (Siamese)...")
            time.sleep(1.2)
            st.write("Scanning Anomaly Fields (Autoencoder)...")
            time.sleep(0.8)
            st.write("Evaluating Quantum Phase Entropy (QPE)...")
            time.sleep(1.5)
            
            st.session_state.results = pipeline.verify(st.session_state.ref_image, st.session_state.suspect_image)
            status.update(label="Scanning Complete", state="complete")
    
    res = st.session_state.results
    
    # Verdict Card
    color = "#238636" if res['verdict'] == "AUTHENTIC" else "#da3633"
    st.markdown(f"""
    <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
        <h1 style="color: white; margin: 0; font-family: 'Orbitron';">{res['verdict']}</h1>
        <h3 style="color: rgba(255,255,255,0.8); margin: 5px;">Confidence: {res['confidence']}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Neural Distance", res['similarity_score'], delta_color="inverse")
    m2.metric("Visual Anomaly", res['anomaly_score'], delta_color="inverse")
    m3.metric("Quantum Fidelity", res['quantum_score'])

    with st.expander("🔬 View Quantum Spectral Signature"):
        q_details = res.get('quantum_details', {})
        counts = q_details.get('counts', {})
        top_phase = q_details.get('top_phase', 0.0)
        
        # Plotly Circle Plot
        fig = go.Figure()
        fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line_color="white", opacity=0.3)
        rad = 2 * math.pi * top_phase
        fig.add_trace(go.Scatter(x=[0, math.cos(rad)], y=[0, math.sin(rad)], mode='lines+markers', 
                                 marker=dict(size=12, color='#00d4ff'), line=dict(color='#00d4ff', width=5), name="Phase"))
        fig.update_layout(width=300, height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The unique spectral phase identified is **{top_phase:.4f}π**. Original artworks typically exhibit a phase of 0.5π.")

    col1, col2 = st.columns(2)
    if col1.button("← NEW SCAN"):
        reset_system()
    if col2.button("PROCEED TO FEEDBACK →"):
        next_step()
        st.rerun()

# --- STEP 4: FEEDBACK & EVOLUTION ---
elif st.session_state.step == 4:
    st.markdown("<div class='step-header'>STEP 4: TRAIN THE EVOLVED MIND</div>", unsafe_allow_html=True)
    st.write("Help AURA learn. If the AI was wrong, provide the correct label to retrain the quantum kernel.")
    
    res = st.session_state.results
    st.markdown(f"**Current System Verdict:** `{res['verdict']}`")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.suspect_image, caption="Analyzed Work", use_container_width=True)
    
    with col2:
        st.subheader("Teacher Input")
        is_correct = st.radio("Was the result correct?", ["Yes, it is correct", "No, it is an error"])
        
        if is_correct == "No, it is an error":
             correction = st.selectbox("What is the TRUE identity?", ["Authentic", "Fake"])
             if st.button("TEACH & RETRAIN SYSTEM"):
                 with st.status("Evolving Quantum Kernel...", expanded=True):
                     evolution.add_experience(st.session_state.suspect_path, correction)
                     evolution.evolve(pipeline)
                     st.success("AURA has successfully learned from this experience. Its future predictions will be more accurate.")
        else:
            st.success("Excellent. System memory reinforced.")

    st.markdown("---")
    if st.button("CONCLUDE & ISSUE REPORT →"):
        next_step()
        st.rerun()

# --- STEP 5: FINAL REPORT ---
elif st.session_state.step == 5:
    st.markdown("<div class='step-header'>STEP 5: FORENSIC CERTIFICATE</div>", unsafe_allow_html=True)
    
    st.balloons()
    st.success("All analytical protocols concluded. You can now download the official Forensic Verification Report.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(st.session_state.suspect_image, use_container_width=True)
        st.markdown(f"**Case ID:** #{hash(time.time()) % 100000}")
        st.markdown(f"**Final Verdict:** {st.session_state.results['verdict']}")
    
    with col2:
        st.info("Generating PDF forensic document...")
        pdf_bytes = create_pdf_report(st.session_state.ref_image, st.session_state.suspect_image, st.session_state.results)
        
        st.download_button(
            label="💾 DOWNLOAD OFFICIAL FORENSIC REPORT",
            data=pdf_bytes,
            file_name=f"aura_report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        st.markdown("---")
        if st.button("START NEW INVESTIGATION"):
            reset_system()
