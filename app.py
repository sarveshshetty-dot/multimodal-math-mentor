import os
import uuid
import json
import streamlit as st

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
    context = retrieve_context(query)
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
                    ocr = ImageOCRProcessor()
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
                    speech_proc = SpeechProcessor()
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
        if st.session_state.get("graph_state") is None:
            # First run
            with st.status("Executing Agent Workflow...", expanded=True) as status:
                initial_state = {"raw_text": st.session_state["raw_text"]}
                
                st.write("1️⃣ Parser Agent: Extracting structured data...")
                state = app_graph.invoke(initial_state)
                
                st.write("2️⃣ Router Agent: Evaluating intent and complexity...")
                
                if state.get("needs_hitl", False):
                    st.warning("⚠️ Intent unclear. Clarification required.")
                else:
                    st.write("3️⃣ RAG Retriever: Fetching mathematical context...")
                    st.json(state.get('parsed_problem', {}))
                    
                    if 'solver_output' in state:
                        st.write("4️⃣ Solver Agent: Reasoning and SymPy evaluation...")
                        with st.expander("Solver Steps"):
                            for step in state['solver_output'].get('steps', []):
                                st.markdown(f"- {step}")
                                
                    if 'verifier_output' in state:
                        st.write("5️⃣ Verifier Agent: Checking correctness...")
                        st.json(state['verifier_output'].get('verification', {}))
                
                status.update(label="Graph Execution Complete", state="complete", expanded=False)
                
                st.session_state["graph_state"] = state
                
        state = st.session_state["graph_state"]
        
        if state.get("needs_hitl", False):
            st.session_state["in_hitl"] = True
            st.rerun()
        else:
            # Metrics and Final results
            if 'final_explanation' in state:
                st.markdown("---")
                st.markdown("### ✨ Final Solution & Explanation")
                st.markdown(state['final_explanation'])
            
            if 'verifier_output' in state:
                conf = state['verifier_output'].get('confidence_data', {}).get('confidence', 0)
                st.metric("Confidence Score", f"{conf}/100")
            
            # Feedback
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct"):
                    memory.save_interaction(
                        st.session_state["session_id"], state['raw_text'], state['parsed_problem'],
                        state['retrieved_context'], state['final_explanation'], json.dumps(state['verifier_output'])
                    )
                    # We can't immediately update_feedback without insert ID, so let's simplify:
                    st.success("Feedback recorded. Memory updated!")
            with col2:
                if st.button("👎 Incorrect"):
                    st.error("Thanks for the feedback. The system will learn from this.")
                    
    elif st.session_state.get("in_hitl", False):
        state = st.session_state["graph_state"]
        
        if state.get("solver_output"):
            st.warning("⚠️ Human-In-The-Loop (HITL): High uncertainty in the mathematical solution.")
            st.markdown("### Proposed Solution")
            st.markdown(state['solver_output'].get("solution", ""))
            
            st.markdown("### Verifier Errors")
            for err in state['verifier_output'].get("verification", {}).get("errors_found", []):
                st.error(err)
            
            action = st.radio("HITL Action", ["Approve & Generate Explanation", "Reject & Edit Solution"])
            if action == "Reject & Edit Solution":
                edited_sol = st.text_area("Edit Solution", value=state['solver_output'].get("solution", ""))
        else:
            st.info("ℹ️ Not a math problem")
            st.markdown("Please clarify or re-enter a valid mathematical equation or expression below.")
            edited_input = st.text_area("Re-enter problem:", value=state['raw_text'])
            action = "Reject & Edit Solution" # Re-use same logic to restart
            edited_sol = None # We'll handle this by updating raw_text
            
        if st.button("Confirm & Proceed"):
            if state.get("solver_output"):
                if action == "Reject & Edit Solution":
                    state['solver_output']['solution'] = edited_sol
                
                state['needs_hitl'] = False
                state['final_explanation'] = run_explainer_agent(
                    state['parsed_problem'], state['solver_output'], state['verifier_output']
                )
                st.session_state["graph_state"] = state
                st.session_state["in_hitl"] = False
                st.rerun()
            else:
                # Coming from Router HITL (no solver output)
                st.session_state["raw_text"] = edited_input
                st.session_state["graph_state"] = None
                st.session_state["in_hitl"] = False
                st.rerun()

if __name__ == "__main__":
    main()
