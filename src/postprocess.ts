export type LineRect = { x: number; y: number; w: number; h: number };

export function renderLines(canvas: HTMLCanvasElement, lines: LineRect[]) {
  const ctx = canvas.getContext('2d')!;
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#00c';
  lines.forEach((r) => {
    ctx.strokeRect(r.x, r.y, r.w, r.h);
  });
}

export function linesToJSON(lines: LineRect[]): string {
  return JSON.stringify(lines);
}