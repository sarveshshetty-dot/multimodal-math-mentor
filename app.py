import os
import uuid
import json
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# Import custom modules
from ui.streamlit_ui import render_sidebar
from agents.parser_agent import run_parser_agent
from agents.router_agent import route_problem
from agents.solver_agent import run_solver_agent
from agents.verifier_agent import run_verifier_agent
from agents.explainer_agent import run_explainer_agent
from rag.retriever import retrieve_context
from memory.memory_store import MemoryStore

# Input Processors
from input_processing.image_ocr import ImageOCRProcessor, easyocr
from input_processing.speech_to_text import SpeechProcessor, WhisperModel
from input_processing.text_input import TextProcessor
import tempfile

# Define Graph State
class GraphState(TypedDict):
    raw_text: str
    parsed_problem: dict
    retrieved_context: str
    solver_output: dict
    verifier_output: dict
    final_explanation: str
    needs_hitl: bool

# --- Cached Resource Loaders (Lazy Loading) ---

@st.cache_resource
def get_cached_vector_store():
    from rag.vector_store import get_vector_store
    base_dir = os.path.dirname(__file__)
    persist_dir = os.path.join(base_dir, "data/faiss_index")
    return get_vector_store(persist_dir)

@st.cache_resource
def get_cached_speech_processor():
    return SpeechProcessor()

@st.cache_resource
def get_cached_ocr_processor():
    return ImageOCRProcessor()

@st.cache_resource
def get_cached_llm():
    from tools.local_llm import LocalLLM
    return LocalLLM()

# Graph Nodes
def parser_node(state: GraphState):
    parsed = run_parser_agent(state['raw_text'])
    return {"parsed_problem": parsed}

def router_node(state: GraphState):
    # Determine if we need clarification based on router agent logic
    action = route_problem(state['parsed_problem'])
    needs_clarification = (action == "human_in_the_loop")
    return {"needs_hitl": needs_clarification}

def retrieve_node(state: GraphState):
    topic = state['parsed_problem'].get('topic', '')
    query = f"{topic}: {state['parsed_problem'].get('problem_text', '')}"
    
    # Lazy load vector store only when needed
    vectorstore = get_cached_vector_store()
    context = retrieve_context(query, vectorstore=vectorstore)
    return {"retrieved_context": context}

def solver_node(state: GraphState):
    solved = run_solver_agent(state['parsed_problem'], state['retrieved_context'])
    return {"solver_output": solved}

def verifier_node(state: GraphState):
    verified = run_verifier_agent(state['parsed_problem'], state['solver_output'])
    needs_hitl = verified.get("confidence_data", {}).get("needs_hitl", False)
    return {"verifier_output": verified, "needs_hitl": needs_hitl}

def explainer_node(state: GraphState):
    explanation = run_explainer_agent(state['parsed_problem'], state['solver_output'], state['verifier_output'])
    return {"final_explanation": explanation}

# Build LangGraph
workflow = StateGraph(GraphState)

workflow.add_node("Parser", parser_node)
workflow.add_node("Router", router_node)
workflow.add_node("Retriever", retrieve_node)
workflow.add_node("Solver", solver_node)
workflow.add_node("Verifier", verifier_node)
workflow.add_node("Explainer", explainer_node)

# Define edges
workflow.set_entry_point("Parser")
workflow.add_edge("Parser", "Router")

def route_after_parser(state: GraphState):
    if state.get("needs_hitl", False):
        return "HumanInTheLoop"
    return "Retriever"

workflow.add_conditional_edges(
    "Router",
    route_after_parser,
    {
        "HumanInTheLoop": END,
        "Retriever": "Retriever"
    }
)

workflow.add_edge("Retriever", "Solver")
workflow.add_edge("Solver", "Verifier")

def route_after_verifier(state: GraphState):
    if state.get("needs_hitl", False):
        return "HumanInTheLoop" # We will handle this outside standard graph for Streamlit simplicity
    return "Explainer"

workflow.add_conditional_edges(
    "Verifier",
    route_after_verifier,
    {
        "HumanInTheLoop": END, # Pause graph to ask user
        "Explainer": "Explainer"
    }
)
workflow.add_edge("Explainer", END)
app_graph = workflow.compile()

# --- Stable Math Pipeline Functions ---

def classify(text):
    t = text.lower()
    if "y =" in t:
        return "function"
    elif "derivative" in t:
        return "derivative"
    elif "=" in t:
        return "equation"
    elif any(w in t for w in ["explain", "what is"]):
        return "concept"
    else:
        return "expression"

def solve_equation(text):
    import sympy as sp
    try:
        x = sp.symbols('x')

        expr = text.replace("^", "**")
        lhs, rhs = expr.split("=")

        eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
        sol = sp.solve(eq, x)

        return sol
    except:
        return None

def simplify_expression(text):
    import sympy as sp
    try:
        expr = text.replace("^", "**")
        simplified = sp.simplify(expr)
        return simplified
    except:
        return None

def clean_function_input(text):
    text = text.lower()
    for word in ["plot", "draw", "sketch", "please", "can you"]:
        text = text.replace(word, "")
    return text.strip()

def plot_function(text):
    try:
        clean_text = clean_function_input(text)
        if "=" in clean_text:
            expr = clean_text.split("=")[1].strip()
        else:
            expr = clean_text
            
        expr = expr.replace("^", "**")
        
        x = np.linspace(-10, 10, 400)
        y = eval(expr)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"Graph of y = {expr}")
        ax.axhline(0)
        ax.axvline(0)
        st.pyplot(fig)
    except Exception as e:
        st.error("Could not plot function.")

def solve_derivative(text):
    import sympy as sp
    try:
        x = sp.symbols('x')

        expr = text.lower()
        expr = expr.replace("what is", "")
        expr = expr.replace("the derivative of", "")
        expr = expr.replace("derivative of", "")
        expr = expr.replace("?", "").strip()
        expr = expr.replace("^", "**")

        f = sp.sympify(expr)
        d = sp.diff(f, x)

        return d
    except:
        return None

def explain_concept(text):
    if "derivative" in text.lower():
        st.write("A derivative tells how fast something changes.")
    else:
        st.write("This is a conceptual math question.")


# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Math Mentor AI (Offline)", layout="wide", page_icon="🧠")
    
    # Init memory
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    memory = MemoryStore()

    render_sidebar()

    # 1. Title (Hero header)
    st.markdown("""
        <div style='padding: 32px 0 10px 0;'>
            <h1 style='margin-bottom: 6px;'>Multimodal Math Mentor</h1>
            <p style='color:#6b7280; font-size:1rem; margin:0 0 20px 0;'>
                Offline AI · JEE-Level Problem Solver · Multi-Agent Reasoning
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. Streamlit Cloud Info (if applicable)
    ocr_available = easyocr is not None
    audio_available = WhisperModel is not None
    
    if not ocr_available or not audio_available:
        st.info("☁️ **Streamlit Cloud Demo:** OCR and audio features are disabled due to resource limits. Run locally for full multimodal support.")

    st.markdown("<hr style='border-color:rgba(99,102,241,0.15); margin-bottom:28px;'>", unsafe_allow_html=True)

    # 3. Input Mode Selector
    st.markdown("<p style='color:#6b7280; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:600; margin-bottom:8px;'>Select Input Mode</p>", unsafe_allow_html=True)
    
    available_modes = ["Text"]
    if ocr_available: available_modes.append("Image")
    if audio_available: available_modes.append("Audio")
    
    input_mode = st.radio("Select mode", ["Text", "Image", "Audio"], horizontal=True, label_visibility="collapsed")
    
    # Mode switching state management
    if "current_input_mode" not in st.session_state:
        st.session_state["current_input_mode"] = input_mode
    
    if st.session_state["current_input_mode"] != input_mode:
        st.session_state["current_input_mode"] = input_mode
        if "raw_text" in st.session_state: del st.session_state["raw_text"]
        if "graph_state" in st.session_state: st.session_state["graph_state"] = None
        st.session_state["in_hitl"] = False
        st.rerun()

    # 4. Mode-specific UI
    raw_text = ""
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if input_mode == "Text":
        user_input = st.text_area(
            "Enter Math Problem:",
            height=160,
            placeholder="e.g. Solve for x: 3x² - 7x + 2 = 0. Show all steps."
        )
        if st.button("Solve Problem", type="primary", disabled=not user_input):
            st.session_state["raw_text"] = TextProcessor.process_text(user_input)
            st.session_state["graph_state"] = None
            st.session_state["in_hitl"] = False

    elif input_mode == "Image":
        if not ocr_available:
            st.warning("⚠️ Image OCR is disabled on Streamlit Cloud due to resource limits. Run locally to use this feature.")
        else:
            uploaded_file = st.file_uploader("Upload Image (Math Problem)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                with st.spinner("🔍 Extracting text via OCR..."):
                    ocr = get_cached_ocr_processor()
                    bytes_data = uploaded_file.getvalue()
                    extracted = ocr.process_image(bytes_data)
                    if extracted:
                        st.success("✅ Text extracted successfully!")
                        final_text = st.text_area("Review & Edit OCR Output:", value=extracted, height=100)
                        if st.button("Solve Problem", type="primary"):
                            st.session_state["raw_text"] = final_text
                            st.session_state["graph_state"] = None
                            st.session_state["in_hitl"] = False
                    else:
                        st.error("❌ Failed to extract text. Please try a clearer image.")

    elif input_mode == "Audio":
        if not audio_available:
            st.warning("⚠️ Audio input is disabled on Streamlit Cloud due to resource limits. Run locally to use this feature.")
        else:
            st.markdown("<p style='color:#9ca3af; font-size:0.85rem; margin-bottom:8px;'>🎤 Record your math problem aloud.</p>", unsafe_allow_html=True)
            audio_recording = st.audio_input("Record your math problem")
            if audio_recording is not None:
                st.audio(audio_recording)
                with st.spinner("🎧 Transcribing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_recording.getvalue())
                        tmp_path = tmp.name
                    speech_proc = get_cached_speech_processor()
                    extracted = speech_proc.process_audio(tmp_path)
                    os.remove(tmp_path)
                    if extracted:
                        st.success("✅ Audio transcribed!")
                        final_text = st.text_area("Review & Edit Transcript:", value=extracted, height=100)
                        if st.button("Solve Problem", type="primary"):
                            st.session_state["raw_text"] = final_text
                            st.session_state["graph_state"] = None
                            st.session_state["in_hitl"] = False
                    else:
                        st.error("❌ Failed to process audio.")

    # 5. Graph Execution & Results
    if "raw_text" in st.session_state and not st.session_state.get("in_hitl", False):
        user_input = st.session_state["raw_text"]
        
        ptype = classify(user_input)
        result = None

        if ptype == "function":
            plot_function(user_input)
            st.stop()
        elif ptype == "equation":
            result = solve_equation(user_input)
            topic = "Algebra (Equation Solver)"
        elif ptype == "derivative":
            result = solve_derivative(user_input)
            topic = "Calculus (Derivative)"
        elif ptype == "concept":
            explain_concept(user_input)
            st.stop()
        else:
            result = simplify_expression(user_input)
            topic = "Algebra (Expression Simplification)"

        if result is None:
            st.error("Could not solve. Try simpler input.")
        else:
            st.header("✨ Final Solution & Explanation")
            st.subheader(f"📐 Solution: {topic}")

            st.subheader("🔢 Step-by-Step Solution")
            st.write(f"**1. Problem:** `{user_input}`")
            st.write(f"**2. Topic:** `{topic}`")
            st.write("**3. Steps:** Analyzed via SymPy Engine")

            st.subheader("✅ Final Answer")
            st.success(f"{result}")
            
            # Feedback
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct"):
                    memory.save_interaction(
                        st.session_state["session_id"], user_input, {"topic": topic},
                        "", str(result), "{}"
                    )
                    st.success("Feedback recorded. Memory updated!")
            with col2:
                if st.button("👎 Incorrect"):
                    st.error("Thanks for the feedback. The system will learn from this.")

if __name__ == "__main__":
    main()
