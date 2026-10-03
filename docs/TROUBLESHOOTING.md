# Troubleshooting

## Playback

| Problem | Fix |
|---|---|
| No sound with `roomeq run` | The system output must be **BlackHole 2ch** while RoomEQ runs. It switches automatically with `switchaudio-osx`; otherwise set it in System Settings → Sound. |
| Silence after quitting | Switch the output back to your speakers (automatic with `switchaudio-osx`). |
| Crackles | `--blocksize 512`. The glitch counter (terminal and dashboard) shows whether the buffer ran dry. |
| Volume keys greyed out | Expected with BlackHole as the output: use the speaker's knob/remote or `+`/`-`. |
| Lip-sync on video | `--blocksize 128` (about 30 ms instead of 38 ms). |
| Dashboard says "engine not reachable" | `roomeq run` isn't running, or port 8080 is taken: change `server.http_port`. |

## Phone connection

| Problem | Fix |
|---|---|
| Page shows the certificate steps every time | The switch in Certificate Trust Settings is still off. |
| "This Connection Is Not Private" | You opened the HTTPS address before trusting the certificate. Open the QR link (`http://…:8080`). |
| Timed out waiting for the phone | Same Wi-Fi? Allow Python in System Settings → Network → Firewall, or use `--tunnel`. |
| "Phone audio stalled" | The screen locked or Safari went to the background. |
| "The phone browser kept echoCancellation … on" | Use Safari (not an in-app browser) and keep iOS up to date. |

## Measurement

| Message | Meaning and fix |
|---|---|
| Could not find both sync markers | The phone didn't record the whole signal, or the volume is far too low. |
| Recording clipped | Lower the volume; moderate listening level is plenty. |
| Low signal-to-noise | Raise the volume; switch off fans/air conditioning. |
| Repeated sweeps disagree | Something moved or made noise during a sweep. Measure again. |
| Unusually large clock drift | Treated as unreliable and retaken automatically. |
| Bass stays N dB above the target | Turn the subwoofer down on the speaker by about N dB, then auto-tune again. |
| Auto-tune measured much worse than predicted | Run `roomeq verify` at your listening volume. It tells apart a problem in the EQ path, a speaker with level-dependent bass processing, and plain position variance. |
| Deep narrow dip left alone | A room null (cancellation). EQ cannot fill it; moving the seat or subwoofer can. |
