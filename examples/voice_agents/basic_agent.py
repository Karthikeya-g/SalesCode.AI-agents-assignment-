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
from livekit.plugins import silero, deepgram, openai

logger = logging.getLogger("basic-agent")
load_dotenv()

# --- REQ[cite: 37, 78]: CONFIGURABLE IGNORE LIST ---
# Modular list that can be easily edited.
# We include variations of backchannel words to handle STT instability.
IGNORE_WORDS = {
    "yeah", "yep", "yes", "ok", "okay", "alright", 
    "hmm", "mhm", "aha", "uh-huh", "uhhuh", "mm-hm", 
    "right", "sure", "correct", "yea", "oka"
}

# --- REQ: MODULAR LOGIC CLASS ---
class InterruptionHandler:
    """
    Handles the logic layer for distinguishing between backchanneling
    and active interruptions.
    """
    @staticmethod
    def is_backchannel(text: str) -> bool:
        """
        Returns True if the input text contains ONLY words from the ignore list.
        """
        # Clean text: Remove punctuation and hyphens to standardize "uh-huh" -> "uh huh"
        clean_text = text.replace("-", " ").translate(str.maketrans('', '', string.punctuation)).lower()
        words = clean_text.split()
        
        if not words:
            return True # Treat empty noise as safe
            
        # Check if ALL words are in the ignore list
        # Logic: If set(words) is a subset of IGNORE_WORDS, it's a backchannel.
        return set(words).issubset(IGNORE_WORDS)

    @staticmethod
    def should_interrupt(text: str, is_agent_speaking: bool) -> bool:
        """
        Decides if the agent should stop speaking based on the logic matrix.
        REQ[cite: 16]: Logic Matrix Implementation.
        """
        if not is_agent_speaking:
            return False # REQ[cite: 38]: Filter applies only when generating audio
        
        # If it is a backchannel (e.g., "Yeah"), DO NOT interrupt.
        if InterruptionHandler.is_backchannel(text):
            return False 
            
        # REQ[cite: 39]: Semantic Interruption. 
        # If it's NOT a backchannel (contains other words like "Wait"), interrupt.
        return True

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You are curious, friendly, and have a sense of humor. "
            "Keep responses concise. Speak English.",
        )

    async def on_enter(self):
        await self.session.generate_reply()

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # --- REQ[cite: 45]: Integration with existing framework ---
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        vad=ctx.proc.userdata["vad"],
        
        # --- REQ[cite: 13, 41]: STRICT REQUIREMENT & NO VAD MOD ---
        # We disable automatic turn detection to prevent the VAD from 
        # stopping the agent on "Yeah". We will handle turns manually.
        turn_detection=None,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # --- REQ[cite: 46, 47]: TRANSCRIPTION LOGIC & FALSE START STRATEGY ---
    @session.on("user_transcription_received")
    def on_transcription(msg):
        """
        Stream handler to decide interruption in real-time (Low Latency REQ [cite: 49]).
        """
        text = msg.content.strip()
        if not text: return

        is_speaking = session.response_agent.is_speaking
        
        # Use our modular handler to decide
        if InterruptionHandler.should_interrupt(text, is_speaking):
            logger.info(f"🔴 INTERRUPT: Valid command detected -> '{text}'")
            session.response_agent.interrupt()
        elif is_speaking:
            # REQ[cite: 32]: IGNORE behavior
            logger.info(f"🟢 IGNORE: Backchannel detected -> '{text}'")

    # --- REQ[cite: 27, 34, 74]: STATE AWARENESS (Silent Response) ---
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        """
        Handler for when the user finishes a sentence.
        If the agent is silent, we MUST reply, even if it was just "Yeah".
        """
        # Only reply if the agent isn't currently talking (normal turn-taking)
        if not session.response_agent.is_speaking:
            text = msg.content.strip()
            logger.info(f"🔵 REPLY: User finished turn ('{text}'). Generating reply.")
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
