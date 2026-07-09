"""Avatar session state machine (idle / listening / speaking).

Pure and clock-free so it unit-tests anywhere. Transitions mirror the
LiveAvatar LITE turn cycle the OAV spec documents:

- `agent.speak` chunks (same event_id per utterance) put the avatar in
  SPEAKING; frames come from talking clips re-lipped against the live PCM.
- `agent.speak_end` marks the utterance complete; the avatar keeps speaking
  until buffered audio drains, then returns to IDLE.
- `agent.interrupt` flushes audio and crossfades to LISTENING.
- `agent.start_listening` / `agent.stop_listening` toggle LISTENING while
  the avatar is not speaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AvatarState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class Transition:
    previous: AvatarState
    current: AvatarState
    reason: str


@dataclass
class AvatarStateMachine:
    state: AvatarState = AvatarState.IDLE
    active_event_id: str | None = None
    # True once speak_end arrived for the active utterance; the session
    # returns to idle when buffered audio drains.
    end_pending: bool = False
    interrupts: int = 0
    utterances: int = 0
    transitions: list[Transition] = field(default_factory=list)

    def _move(self, target: AvatarState, reason: str) -> Transition | None:
        if target == self.state:
            return None
        transition = Transition(self.state, target, reason)
        self.state = target
        self.transitions.append(transition)
        return transition

    def on_speak(self, event_id: str) -> Transition | None:
        """A `agent.speak` PCM chunk arrived for `event_id`."""
        if event_id != self.active_event_id:
            self.active_event_id = event_id
            self.utterances += 1
            self.end_pending = False
        return self._move(AvatarState.SPEAKING, "speak")

    def on_speak_end(self, event_id: str | None = None) -> None:
        """`agent.speak_end` — finish the utterance once audio drains."""
        if event_id is not None and event_id != self.active_event_id:
            return
        if self.state is AvatarState.SPEAKING:
            self.end_pending = True

    def on_audio_drained(self) -> Transition | None:
        """The scheduler ran out of buffered speech audio."""
        if self.state is AvatarState.SPEAKING and self.end_pending:
            self.end_pending = False
            self.active_event_id = None
            return self._move(AvatarState.IDLE, "speak_drained")
        return None

    def on_interrupt(self) -> Transition | None:
        """`agent.interrupt` — cut speech, crossfade to listening."""
        if self.state is not AvatarState.SPEAKING:
            return None
        self.interrupts += 1
        self.active_event_id = None
        self.end_pending = False
        return self._move(AvatarState.LISTENING, "interrupt")

    def on_start_listening(self) -> Transition | None:
        if self.state is AvatarState.SPEAKING:
            # Speech owns the frame until speak_end/interrupt.
            return None
        return self._move(AvatarState.LISTENING, "start_listening")

    def on_stop_listening(self) -> Transition | None:
        if self.state is not AvatarState.LISTENING:
            return None
        return self._move(AvatarState.IDLE, "stop_listening")
