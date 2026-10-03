// Collects raw microphone samples in 2048-frame chunks and hands them to the main thread.
// Missing input (e.g. a muted track) is sent as silence so sample indices stay continuous.
class RoomEQRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 2048;
    this.buf = new Float32Array(this.size);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const ch = input && input.length ? input[0] : null;
    const frames = ch ? ch.length : 128;
    for (let i = 0; i < frames; i++) {
      this.buf[this.n++] = ch ? ch[i] : 0;
      if (this.n === this.size) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("roomeq-recorder", RoomEQRecorder);
