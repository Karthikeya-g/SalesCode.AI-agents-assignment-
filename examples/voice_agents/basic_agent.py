import logging
import string
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero

# NOTE: We removed MultilingualModel from imports because we are handling turns manually
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")
load_dotenv()

# --- 1. CONFIGURATION: IGNORE LIST ---
IGNORE_WORDS = {
    "yeah", "yep", "yes", "ok", "okay", "alright", 
    "hmm", "mhm", "aha", "uh-huh", "right", "sure",
    "yea", "oka", "uh-huh", "uhhuh", "mhm", "mm-hm", "mmhm"
}

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You are curious, friendly, and have a sense of humor. "
            "Keep responses concise. Speak English.",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # --- 2. INITIALIZE SESSION (MANUAL MODE) ---
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=ctx.proc.userdata["vad"],
        
        # CRITICAL CHANGE: Set turn_detection to None.
        # This disables the default "Stop on Speech" behavior.
        # We will handle interruptions manually below.
        turn_detection=None, 
        
        # We don't need preemptive generation if we control the flow manually
        preemptive_generation=False,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # --- 3. LOGIC LAYER: HANDLING INTERRUPTIONS ---
    @session.on("user_transcription_received")
    def on_transcription(msg):
        text = msg.content.strip()
        if not text: return

        # ROBUST CLEANING: 
        # 1. Replace hyphens with spaces (fixes "uh-huh" -> "uh huh")
        # 2. Remove other punctuation
        clean_text = text.replace("-", " ").translate(str.maketrans('', '', string.punctuation)).lower()
        words = clean_text.split()

        # Check for command
        is_real_command = any(word not in IGNORE_WORDS for word in words)

        if session.response_agent.is_speaking:
            if is_real_command:
                print(f"🔴 INTERRUPT: '{text}'")
                session.response_agent.interrupt()
            else:
                # This log is crucial for debugging your assignment!
                print(f"🟢 IGNORE: Backchannel detected '{text}'")

    # --- 4. LOGIC LAYER: HANDLING REPLIES ---
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        # This runs when the user FINISHES a sentence.
        # Since we disabled turn_detection, we must manually tell the agent to reply.
        
        # If the agent is NOT speaking, it means it's the user's turn.
        if not session.response_agent.is_speaking:
            logger.info("🔵 REPLY: User finished speaking, generating reply.")
            # We trigger the reply manually
            session.generate_reply()

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

if __name__ == "__main__":
    cli.run_app(server)
