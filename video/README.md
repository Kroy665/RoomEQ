# Walkthrough video

`docs/media/roomeq-walkthrough.mp4` is generated, not screen-recorded by hand:

| File | Role |
|---|---|
| `script.json` | narration, one entry per segment |
| `actions.mjs` | what the recorder does on screen during each segment (Playwright) |
| `demo-home/presets/Living_room_measured.json` | the real measured preset shown in the "real room" segment |

The take is recorded against `roomeq demo` (real engine/solver/dashboard, simulated room and phone), so
it needs no audio hardware and every auto-tune and verify in the video runs for real.

## Regenerate

Requires the `app-walkthrough-video` tooling (`setup.sh`: Kokoro TTS, ffmpeg) and Playwright in this
folder (`npm i && npx playwright install chromium`). `$SKILL` is that skill's directory.

```bash
PY=~/.cache/walkthrough-video/venv/bin/python

# 1. voice-over (Kokoro af_heart)
$PY $SKILL/scripts/tts.py --script video/script.json --build video/build

# 2. a clean demo server on a spare port, with one auto-tune already run (so the intro shows a result)
rm -rf video/demo-home/measurements; find video/demo-home/presets -name 'autotune*' -delete
ROOMEQ_HOME=$PWD/video/demo-home uv run roomeq demo --port 3200 &
curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"kind":"autotune","positions":3,"repeats":1,"iterations":2}' localhost:3200/api/job/start
# wait ~40 s for it to finish

# 3. record (phone_page opens the LAN address so the first-visit certificate steps show)
cd video && ROOMEQ_LAN=http://<mac-lan-ip>:3200 node $SKILL/scripts/capture.mjs \
     --script script.json --actions actions.mjs --build build --base http://localhost:3200

# 4. assemble
$PY $SKILL/scripts/assemble.py --script script.json --build build --out out/roomeq-walkthrough.mp4
cp out/roomeq-walkthrough.mp4 ../docs/media/
```

`build/` and `out/` are not committed.
