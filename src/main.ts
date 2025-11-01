import { renderLines, linesToJSON, type LineRect } from './postprocess';
const worker = new Worker(new URL('./worker/dbnet.worker.ts', import.meta.url));

// Global DnD fallback: allow dropping anywhere, even if UI isn't ready yet
window.addEventListener('dragover', (ev) => { ev.preventDefault(); });
window.addEventListener('drop', async (ev) => {
  ev.preventDefault();
  const dt = ev.dataTransfer;
  if (!dt) return;
  const file = dt.files && dt.files[0];
  if (!file) return;
  await handleFileGlobal(file);
});

async function handleFileGlobal(f: File) {
  let canvasEl = document.getElementById('canvas') as HTMLCanvasElement | null;
  if (!canvasEl) {
    canvasEl = document.createElement('canvas');
    canvasEl.id = 'canvas';
    document.body.appendChild(canvasEl);
  }
  const ctx2d = canvasEl.getContext('2d', { willReadFrequently: true } as any) as CanvasRenderingContext2D;
  const img = await createImageBitmap(f);
  canvasEl.width = img.width;
  canvasEl.height = img.height;
  ctx2d.clearRect(0, 0, img.width, img.height);
  ctx2d.drawImage(img, 0, 0);
  const imageData = ctx2d.getImageData(0, 0, img.width, img.height);
  worker.postMessage({ msg: 'detect', payload: { image: imageData } }, [imageData.data.buffer]);
}

window.addEventListener('DOMContentLoaded', () => {
  const file = document.getElementById('file') as HTMLInputElement | null;
  const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
  const dropzone = document.getElementById('dropzone') as HTMLDivElement | null;

  if (!file || !canvas) {
    console.error('UI elements not found: file or canvas');
    return;
  }

  const canvasEl = canvas as HTMLCanvasElement;
  const ctx2d = canvasEl.getContext('2d') as CanvasRenderingContext2D;
  if (!ctx2d) {
    console.error('2D context unavailable');
    return;
  }

  // worker is created globally

  let W = 0, H = 0;

  worker.onmessage = (e) => {
    const { msg, payload } = e.data;
    if (msg === 'lines') {
      renderLines(canvasEl, payload.lines as LineRect[]);
      const jsonEl = document.getElementById('json') as HTMLPreElement | null;
      if (jsonEl) {
        const json = linesToJSON(payload.lines as LineRect[]);
        jsonEl.textContent = json;
      }
      // const rois: ImageData[] = payload.lines.map((r: LineRect) => cropLine(r));
    } else if (msg === 'error') {
      console.error('Worker error:', payload);
    }
  };

  file.onchange = async () => {
    const f = file.files?.[0];
    if (!f) return;
    await handleFile(f);
  };

  async function handleFile(f: File) {
    const img = await createImageBitmap(f);
    canvasEl.width = W = img.width;
    canvasEl.height = H = img.height;
    ctx2d.clearRect(0, 0, W, H);
    ctx2d.drawImage(img, 0, 0);

    const imageData = ctx2d.getImageData(0, 0, W, H);
    worker.postMessage({
      msg: 'detect',
      payload: { image: imageData },
    }, [imageData.data.buffer]);
  }

  // Drag & Drop support on dropzone and canvas
  const dndTargets: HTMLElement[] = [canvas, ...(dropzone ? [dropzone] : [])];
  dndTargets.forEach(el => {
    el.addEventListener('dragover', (ev) => { ev.preventDefault(); });
    el.addEventListener('dragenter', (ev) => { ev.preventDefault(); });
    el.addEventListener('drop', async (ev) => {
      ev.preventDefault();
      const dt = ev.dataTransfer;
      if (!dt) return;
      const file = dt.files && dt.files[0];
      if (!file) return;
      await handleFile(file);
    });
  });

  function cropLine(rect: LineRect): ImageData {
    const x = Math.max(0, Math.floor(rect.x));
    const y = Math.max(0, Math.floor(rect.y));
    const w = Math.max(1, Math.floor(rect.w));
    const h = Math.max(1, Math.floor(rect.h));
    return ctx2d.getImageData(x, y, w, h);
  }

  (window as any).cropLine = cropLine;
});
