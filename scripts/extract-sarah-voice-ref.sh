#!/usr/bin/env bash
# OAV-3: extract the owned Sarah voice-clone reference wavs from the footage
# bucket and publish them to the voice-ref path. Idempotent; re-run to rebuild.
#
# Requires: gsutil (authed for openagentsgemini), ffmpeg.
# See docs/oav3-tts.md for the transcript pairing and selection evidence.
set -euo pipefail

FOOTAGE="gs://openagentsgemini-oa-artifacts/sarah-avatar/footage"
VOICE_REF="gs://openagentsgemini-oa-artifacts/sarah-avatar/voice-ref"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Clip 4: first utterance, STT transcript confidence 1.00.
CLIP4="v2_traced-4c7b99f9-ed57-41b1-85f2-5cecf550b1e7.mp4"
# Clip 0: single utterance, STT transcript confidence 0.97.
CLIP0="v2_1783572761179-360477728.mp4"
# NOTE: the footage README's "loudest clip" (v2_traced-df7d3b47...) contains
# NO recognizable speech per Google STT; do not use it as a voice reference.

cd "$WORKDIR"
gsutil -q cp "$FOOTAGE/$CLIP4" clip4.mp4
gsutil -q cp "$FOOTAGE/$CLIP0" clip0.mp4

# Primary single-take reference (5.40 s, 24 kHz mono s16le).
ffmpeg -y -loglevel error -i clip4.mp4 -vn -ss 0 -to 5.40 \
  -ac 1 -ar 24000 -c:a pcm_s16le sarah_voice_ref_v1.wav

# Alternate long reference (~11.5 s): clip 4 utterance + clip 0 utterance.
ffmpeg -y -loglevel error -i clip0.mp4 -vn -ss 0 -to 6.10 \
  -ac 1 -ar 24000 -c:a pcm_s16le ref_part0.wav
ffmpeg -y -loglevel error -i sarah_voice_ref_v1.wav -i ref_part0.wav \
  -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" \
  -ar 24000 -ac 1 -c:a pcm_s16le sarah_voice_ref_v1_long.wav

gsutil cp sarah_voice_ref_v1.wav sarah_voice_ref_v1_long.wav "$VOICE_REF/"
echo "published:"
gsutil ls -l "$VOICE_REF/"
