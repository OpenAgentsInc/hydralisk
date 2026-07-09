from __future__ import annotations

from hydralisk.avatar.state import AvatarState, AvatarStateMachine


def test_initial_state_is_idle() -> None:
    machine = AvatarStateMachine()
    assert machine.state is AvatarState.IDLE
    assert machine.utterances == 0
    assert machine.interrupts == 0


def test_speak_cycle_returns_to_idle_after_drain() -> None:
    machine = AvatarStateMachine()

    transition = machine.on_speak("utt-1")
    assert transition is not None
    assert machine.state is AvatarState.SPEAKING
    assert machine.utterances == 1

    # More chunks of the same utterance: no new transition, no new utterance.
    assert machine.on_speak("utt-1") is None
    assert machine.utterances == 1

    # speak_end alone does not leave SPEAKING — audio must drain first.
    machine.on_speak_end("utt-1")
    assert machine.state is AvatarState.SPEAKING
    assert machine.end_pending

    transition = machine.on_audio_drained()
    assert transition is not None
    assert machine.state is AvatarState.IDLE
    assert machine.active_event_id is None


def test_new_event_id_counts_a_new_utterance() -> None:
    machine = AvatarStateMachine()
    machine.on_speak("utt-1")
    machine.on_speak("utt-2")
    assert machine.utterances == 2
    assert machine.active_event_id == "utt-2"


def test_speak_end_for_stale_event_is_ignored() -> None:
    machine = AvatarStateMachine()
    machine.on_speak("utt-2")
    machine.on_speak_end("utt-1")
    assert not machine.end_pending


def test_interrupt_moves_to_listening_and_counts() -> None:
    machine = AvatarStateMachine()
    machine.on_speak("utt-1")
    transition = machine.on_interrupt()
    assert transition is not None
    assert transition.reason == "interrupt"
    assert machine.state is AvatarState.LISTENING
    assert machine.interrupts == 1
    assert machine.active_event_id is None

    # Interrupt outside SPEAKING is a no-op.
    assert machine.on_interrupt() is None
    assert machine.interrupts == 1


def test_listening_toggles_from_idle_only() -> None:
    machine = AvatarStateMachine()
    assert machine.on_start_listening() is not None
    assert machine.state is AvatarState.LISTENING
    assert machine.on_stop_listening() is not None
    assert machine.state is AvatarState.IDLE

    # While speaking, listening controls do not steal the frame.
    machine.on_speak("utt-1")
    assert machine.on_start_listening() is None
    assert machine.state is AvatarState.SPEAKING
    assert machine.on_stop_listening() is None


def test_audio_drained_without_end_pending_keeps_speaking() -> None:
    machine = AvatarStateMachine()
    machine.on_speak("utt-1")
    assert machine.on_audio_drained() is None
    assert machine.state is AvatarState.SPEAKING


def test_transitions_are_recorded() -> None:
    machine = AvatarStateMachine()
    machine.on_speak("utt-1")
    machine.on_interrupt()
    machine.on_stop_listening()
    reasons = [t.reason for t in machine.transitions]
    assert reasons == ["speak", "interrupt", "stop_listening"]
