import streamlit as st
import cv2
from datetime import datetime

from core.brain import AkshuBrain
from core.memory import get_memories
from core.camera_bridge import CameraBridge


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AKSHU AI 2.0",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FUTURISTIC UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(0,180,255,0.08), transparent 30%),
        radial-gradient(circle at 80% 80%, rgba(120,0,255,0.08), transparent 30%),
        #05070d;
    color: #e8f7ff;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

.akshu-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 8px;
    text-align: center;
    margin-bottom: 0;
}

.akshu-subtitle {
    text-align: center;
    color: #7f9db5;
    letter-spacing: 4px;
    font-size: 12px;
    margin-bottom: 25px;
}

.status-card {
    background: rgba(12, 20, 32, 0.85);
    border: 1px solid rgba(0, 180, 255, 0.18);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.status-label {
    color: #71869a;
    font-size: 11px;
    letter-spacing: 2px;
}

.status-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 5px;
}

.panel {
    background: rgba(8, 14, 24, 0.90);
    border: 1px solid rgba(0, 180, 255, 0.16);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 15px;
}

.vision-online {
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    background: rgba(0, 180, 100, 0.10);
    border: 1px solid rgba(0, 220, 130, 0.25);
}

.vision-offline {
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    background: rgba(255, 80, 80, 0.08);
    border: 1px solid rgba(255, 80, 80, 0.20);
}

div[data-testid="stChatMessage"] {
    background: rgba(10, 18, 30, 0.70);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "brain" not in st.session_state:
    st.session_state.brain = AkshuBrain()

if "camera" not in st.session_state:
    st.session_state.camera = CameraBridge()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "AKSHU 2.0 online. "
                "Core systems initialized. "
                "How can I assist you?"
            )
        }
    ]


brain = st.session_state.brain
camera = st.session_state.camera


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="akshu-title">◈ AKSHU AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="akshu-subtitle">'
    'INTELLIGENT COMMAND & VISION SYSTEM • VERSION 2.0'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## AKSHU SYSTEM")

    st.markdown("---")

    st.write("### CORE")

    st.success("● ONLINE")

    st.write("### MEMORY")

    memories = get_memories()

    fact_count = len(memories.get("facts", []))
    project_count = len(memories.get("projects", []))

    st.metric("Stored Facts", fact_count)
    st.metric("Projects", project_count)

    st.markdown("---")

    st.write("### MODULES")

    st.checkbox("AI Brain", value=True, disabled=True)
    st.checkbox("Memory", value=True, disabled=True)
    st.checkbox("YOLO Vision", value=True)
    st.checkbox("OCR", value=True)
    st.checkbox("Face Recognition", value=True)
    st.checkbox("Gesture Detection", value=True)
    st.checkbox("Expression Detection", value=True)

    st.markdown("---")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. AKSHU is ready."
            }
        ]
        st.rerun()


# ============================================================
# STATUS BAR
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-label">CORE</div>
        <div class="status-value">ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">MEMORY</div>
        <div class="status-value">{fact_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">PROJECTS</div>
        <div class="status-value">{project_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">SYSTEM TIME</div>
        <div class="status-value">
            {datetime.now().strftime("%H:%M:%S")}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1.25, 1])


# ============================================================
# COMMAND CENTER
# ============================================================

with left:

    st.markdown("""
    <div class="panel">
        <div class="panel-title">◈ COMMAND CENTER</div>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


# ============================================================
# VISION CENTER
# ============================================================

with right:

    st.markdown("""
    <div class="panel">
        <div class="panel-title">◉ VISION CENTER</div>
    </div>
    """, unsafe_allow_html=True)

    camera_placeholder = st.empty()

    vision_col1, vision_col2 = st.columns(2)

    with vision_col1:

        start_camera = st.button(
            "START CAMERA",
            use_container_width=True
        )

    with vision_col2:

        capture_frame = st.button(
            "SCAN",
            use_container_width=True
        )

    if start_camera:

        if camera.start():

            st.markdown(
                '<div class="vision-online">● CAMERA ONLINE</div>',
                unsafe_allow_html=True
            )

            frame = camera.capture()

            if frame is not None:

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                camera_placeholder.image(
                    frame_rgb,
                    caption="AKSHU Camera Feed",
                    use_container_width=True
                )

        else:

            st.markdown(
                '<div class="vision-offline">'
                '● CAMERA NOT AVAILABLE'
                '</div>',
                unsafe_allow_html=True
            )

    if capture_frame:

        frame = camera.capture()

        if frame is None:

            st.error(
                "Camera unavailable. "
                "Start the camera first."
            )

        else:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            camera_placeholder.image(
                frame_rgb,
                caption="Captured Frame",
                use_container_width=True
            )

            with st.spinner("AKSHU is analyzing the scene..."):

                result = brain.think(
                    "what do you see",
                    image=frame
                )

            st.markdown("### AKSHU ANALYSIS")

            st.info(result["message"])

            if "analysis" in result:

                analysis = result["analysis"]

                results = analysis.get(
                    "results",
                    {}
                )

                objects = results.get(
                    "objects",
                    []
                )

                faces = results.get(
                    "faces",
                    []
                )

                text = results.get(
                    "text",
                    []
                )

                gestures = results.get(
                    "gestures",
                    []
                )

                expressions = results.get(
                    "expressions",
                    []
                )

                m1, m2, m3, m4, m5 = st.columns(5)

                with m1:
                    st.metric(
                        "OBJECTS",
                        len(objects)
                    )

                with m2:
                    st.metric(
                        "FACES",
                        len(faces)
                    )

                with m3:
                    st.metric(
                        "TEXT",
                        len(text)
                    )

                with m4:
                    st.metric(
                        "GESTURES",
                        len(gestures)
                    )

                with m5:
                    st.metric(
                        "EXPRESSIONS",
                        len(expressions)
                    )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Talk to AKSHU..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    result = brain.think(user_input)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["message"]
        }
    )

    st.rerun()
