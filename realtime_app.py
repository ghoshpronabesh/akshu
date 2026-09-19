import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

from core.realtime_vision import RealtimeVisionEngine
from core.voice import AkshuVoice


st.set_page_config(
    page_title="AKSHU 3.0",
    page_icon="??",
    layout="wide"
)


# =========================================================
# AKSHU UI
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(
            circle at top,
            #101a2e 0%,
            #05070d 45%,
            #020308 100%
        );
        color: white;
    }

    .title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: 6px;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #00ffcc;
        letter-spacing: 3px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="title">?? AKSHU 3.0</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">REAL-TIME VISION INTELLIGENCE</div>',
    unsafe_allow_html=True
)


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class AKSHUVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.engine = RealtimeVisionEngine(
            process_every_n_frames=10
        )

        self.received_frames = 0

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.received_frames += 1

        processed = self.engine.process_frame(
            image
        )

        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24"
        )


# =========================================================
# CAMERA
# =========================================================

# AKSHU VOICE ENGINE
if "akshu_voice" not in st.session_state:
    st.session_state.akshu_voice = AkshuVoice()

voice = st.session_state.akshu_voice

ctx = webrtc_streamer(

    key="akshu-realtime-camera-v6",

    video_processor_factory=AKSHUVideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=False
)


# =========================================================
# LIVE STATUS
# =========================================================

st.divider()

st.markdown("### ?? AKSHU LIVE STATUS")


if ctx.video_processor:

    engine = ctx.video_processor.engine

    status = engine.get_status()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "OBJECTS",
            status["objects"]
        )

    with col2:
        st.metric(
            "FACES",
            status["faces"]
        )

    with col3:
        st.metric(
            "AI FPS",
            f'{status["processing_fps"]:.1f}'
        )

    with col4:
        st.metric(
            "AKSHU CORE",
            "ONLINE"
            if status["initialized"]
            else "STARTING"
        )


    # =====================================================
    # SCENE INTELLIGENCE
    # =====================================================

    st.markdown("### ??? SCENE INTELLIGENCE")

    st.info(
        status["summary"]
    )


    analysis = engine.get_latest_analysis()

    results = analysis.get(
        "results",
        {}
    )


    # =====================================================
    # OBJECTS
    # =====================================================

    objects = results.get(
        "objects",
        []
    )

    if objects:

        st.markdown(
            "### ?? DETECTED OBJECTS"
        )

        for obj in objects:

            name = obj.get(
                "class_name",
                "Unknown"
            )

            confidence = obj.get(
                "confidence",
                0
            )

            distance = obj.get(
                "distance"
            )

            if distance is not None:

                st.write(
                    f"**{name}** • "
                    f"Confidence: {confidence:.2f} • "
                    f"Distance: {distance:.2f} m"
                )

            else:

                st.write(
                    f"**{name}** • "
                    f"Confidence: {confidence:.2f}"
                )


    # =====================================================
    # FACES
    # =====================================================

    faces = results.get(
        "faces",
        []
    )

    st.markdown("### ?? FACE INTELLIGENCE")

    if faces:

        st.success(
            f"AKSHU detects {len(faces)} face(s)."
        )

    else:

        st.info(
            "No face detected."
        )


    # =====================================================
    # OCR
    # =====================================================

    text_items = results.get(
        "text",
        []
    )

    st.markdown("### ?? OCR INTELLIGENCE")

    if text_items:

        for text in text_items:

            st.write(
                f"**{text}**"
            )

    else:

        st.info(
            "No recently detected text."
        )


# =========================================================
# ASK AKSHU
# =========================================================

st.divider()

st.markdown(
    "### ?? ASK AKSHU"
)

question = st.text_input(
    "Ask something about the camera scene",
    placeholder="Example: What do you see?"
)


if st.button(
    "?? ASK AKSHU",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not ctx.video_processor:

        st.warning(
            "Start the camera first."
        )

    else:

        engine = ctx.video_processor.engine

        answer = engine.smart_vision.answer(
            question
        )

        st.success(
            f"AKSHU: {answer}"
        )


# =========================================================


# TALK TO AKSHU
st.markdown("## TALK TO AKSHU")

st.caption(
    "Speak a command and AKSHU will process it using "
    "the Brain, Memory and live Vision systems."
)

if "voice_result" not in st.session_state:
    st.session_state.voice_result = None

if st.button(
    "START VOICE COMMAND",
    use_container_width=True,
    key="voice_command_button"
):

    with st.spinner("AKSHU is listening..."):

        command = voice.listen(
            timeout=5,
            phrase_time_limit=8
        )

    if command:

        st.session_state.voice_result = {
            "command": command,
            "speak": True
        }

    else:

        st.session_state.voice_result = {
            "command": None,
            "speak": False
        }

voice_result = st.session_state.voice_result

if voice_result:

    command = voice_result.get("command")

    if command:

        st.success(f"You said: {command}")

        vision_words = [
            "see",
            "camera",
            "object",
            "person",
            "people",
            "face",
            "faces",
            "text",
            "read",
            "gesture",
            "expression",
            "emotion",
            "nearest",
            "closest"
        ]

        try:

            # Get the LIVE vision engine from the active WebRTC processor.
            live_engine = None

            if ctx.video_processor:
                live_engine = ctx.video_processor.engine

            if live_engine is None:
                answer = (
                    "The live vision engine is not active yet. "
                    "Please start the camera first."
                )

            elif any(
                word in command.lower()
                for word in vision_words
            ):

                answer = live_engine.smart_vision.answer(command)

            else:

                brain = st.session_state.get("akshu_brain")

                if brain is None:
                    from core.brain import AkshuBrain
                    brain = AkshuBrain()
                    st.session_state.akshu_brain = brain

                result = brain.think(command)

                answer = result.get(
                    "message",
                    "I have nothing to report."
                )

        except Exception as e:

            answer = f"AKSHU processing error: {e}"

        st.markdown("### AKSHU RESPONSE")
        st.write(answer)

        # Speak ONLY once for the newly captured command.
        if voice_result.get("speak"):

            voice.speak(answer)

            st.session_state.voice_result["speak"] = False

    else:

        st.warning(
            "AKSHU could not understand the command."
        )

        st.session_state.voice_result = None


# FOOTER
# =========================================================

st.divider()

st.caption(
    "AKSHU 3.0 • REAL-TIME COMPUTER VISION • "
    "YOLO + FACE + OCR"
)
