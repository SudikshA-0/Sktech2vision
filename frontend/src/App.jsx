import React, { useEffect, useRef, useState } from 'react';

const iconPaths = {
  home: <><path d="M3 10.5 12 3l9 7.5" /><path d="M5 9v11h14V9M9 20v-7h6v7" /></>,
  pencil: <><path d="m4 20 4.5-1L19 8.5a2.1 2.1 0 0 0-3-3L5.5 16 4 20Z" /><path d="m14.8 6.2 3 3" /></>,
  eraser: <><path d="m7 17-3-3a2 2 0 0 1 0-2.8l6.6-6.6a2 2 0 0 1 2.8 0l6 6a2 2 0 0 1 0 2.8L15 20H8l-1-3Z" /><path d="m8 20 7-7" /></>,
  brush: <><path d="m14 4 6 6M4 20c4 0 6-2 6-5l8-8-3-3-8 8c-3 0-5 2-5 6v2h2Z" /></>,
  shapes: <><rect x="3.5" y="4" width="8" height="7" rx="1" /><circle cx="17" cy="8" r="4" /><path d="m4 20 5-7 5 7H4Z" /></>,
  fill: <><path d="m4 14 7-7 8 8-7 7-8-8Z" /><path d="m11 7 2-2 8 8-2 2M4 20h15" /></>,
  text: <><path d="M4 5h16M12 5v14M8 19h8" /></>,
  undo: <><path d="M9 14 4 9l5-5" /><path d="M4 9h9a7 7 0 0 1 0 14h-2" /></>,
  redo: <><path d="m15 14 5-5-5-5" /><path d="M20 9h-9a7 7 0 0 0 0 14h2" /></>,
  trash: <><path d="M4 7h16M10 11v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3" /></>,
  line: <path d="m5 19 14-14" />,
  rectangle: <rect x="4" y="5" width="16" height="14" rx="1" />,
  circle: <circle cx="12" cy="12" r="8" />,
  triangle: <path d="m12 4 9 16H3L12 4Z" />,
  star: <path d="m12 3 2.8 5.8 6.4.9-4.6 4.5 1.1 6.4-5.7-3-5.7 3 1.1-6.4L2.8 9.7l6.4-.9L12 3Z" />,
  arrow: <><path d="M4 12h15M13 5l7 7-7 7" /></>,
  upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 16v4h16v-4" /></>,
  download: <><path d="M12 4v12m-5-5 5 5 5-5" /><path d="M4 20h16" /></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5.5 9A7 7 0 0 1 18 6l2 2M4 16l2 2a7 7 0 0 0 12.5-3" /></>,
  sparkle: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></>,
  settings: <><circle cx="12" cy="12" r="3" /><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.6a8 8 0 0 1-1.7 1l-.3 1.8h-2.8l-.3-1.8a8 8 0 0 1-1.7-1l-1.7.6-1.4-2.4L7.9 15a8 8 0 0 1 0-2l-1.4-1.1 1.4-2.4 1.7.6a8 8 0 0 1 1.7-1l.3-1.8h2.8l.3 1.8a8 8 0 0 1 1.7 1l1.7-.6 1.4 2.4-1.4 1.1a8 8 0 0 1-.1 2Z" transform="translate(-1 -1) scale(1.08)" /></>,
  image: <><rect x="3" y="4" width="18" height="16" rx="1" /><circle cx="8.5" cy="9" r="1.5" /><path d="m21 15-5-5L5 20" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  chevron: <path d="m9 18 6-6-6-6" />,
  house: <><path d="m3 11 9-7 9 7" /><path d="M5 10v10h14V10M9 20v-6h6v6" /></>,
  car: <><path d="m4 15 1.5-6h13L20 15v4h-2v-2H6v2H4v-4Z" /><path d="m6 9 2-4h8l2 4M7 13h.1M17 13h.1" /></>,
  cat: <><path d="m6 9-2-5 6 3a8 8 0 0 1 4 0l6-3-2 5a8 8 0 1 1-12 0Z" /><path d="M9 12h.1M15 12h.1M10 16h4" /></>,
  chair: <><path d="M7 12V5h10v7M5 12h14v4H5zM7 16v5m10-5v5" /></>,
  bag: <><path d="M5 8h14l1 13H4L5 8Z" /><path d="M9 8V6a3 3 0 0 1 6 0v2" /></>,
};

function Icon({ name, size = 16, className = '' }) {
  return <svg className={className} aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{iconPaths[name]}</svg>;
}

function LogoMark() {
  return <svg className="logo-mark" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="markColor" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#fa4f85"/><stop offset=".48" stopColor="#7b59ee"/><stop offset="1" stopColor="#18a9d6"/></linearGradient></defs><circle cx="15" cy="16" r="13" fill="url(#markColor)"/><path d="m10 21 2-7 9-7 4 4-7 9-8 1Z" fill="#fff" stroke="#303447" strokeWidth="1.4" strokeLinejoin="round"/><path d="m12 14 4 4" stroke="#303447" strokeWidth="1.2"/><path d="m26 3 .7 2.2L29 6l-2.3.8L26 9l-.7-2.2L23 6l2.3-.8L26 3Z" fill="#f5ae2c"/></svg>;
}

function Header({ onUploadClick, onDescribe, isDescribing, canDescribe }) {
  return (
    <header className="topbar">
      <a className="brand" href="#workspace" aria-label="Sketch2Vision home"><LogoMark /><span className="brand-copy"><strong>Sketch2Vision</strong><small>Describe your sketch with AI</small></span></a>
      <nav className="main-nav" aria-label="Main navigation">
        <button className="nav-item active" type="button" aria-current="page"><Icon name="home" size={13} /><span>Draw &amp; Describe</span></button>
        <button className="nav-item" type="button" disabled><Icon name="image" size={13} /><span>Gallery</span></button>
        <button className="nav-item" type="button" onClick={onUploadClick}><Icon name="upload" size={13} /><span>Upload</span></button>
        <button className="nav-item" type="button" disabled><span className="about-dot">i</span><span>About</span></button>
      </nav>
      <div className="header-actions"><button className="header-generate" type="button" disabled={!canDescribe || isDescribing} onClick={onDescribe}>{isDescribing ? <span className="button-spinner" aria-hidden="true" /> : <Icon name="sparkle" size={14} />}<span>{isDescribing ? 'Describing…' : 'Describe Sketch'}</span></button><button className="settings-button" type="button" disabled aria-label="Settings"><Icon name="settings" size={17} /></button></div>
    </header>
  );
}

const palette = [
  ['Black', '#252525'], ['Grey', '#a6a9b0'], ['Silver', '#d2d4d9'], ['Red', '#ef4444'], ['Orange', '#f97316'],
  ['Yellow', '#facc15'], ['Green', '#55bd47'], ['Blue', '#1d9ae2'], ['Purple', '#843de0'], ['Pink', '#ec4899'],
  ['Brown', '#874529'], ['Cyan', '#22c1c3'], ['Violet', '#7065ec'],
];

const shapeTools = [['Line', 'line'], ['Rectangle', 'rectangle'], ['Circle', 'circle'], ['Triangle', 'triangle'], ['Star', 'star'], ['Arrow', 'arrow']];

function Toolbox({ activeTool, setActiveTool, brushSize, setBrushSize, opacity, setOpacity, selectedColor, setSelectedColor, shapesExpanded, setShapesExpanded }) {
  const tools = [['Pencil', 'pencil'], ['Eraser', 'eraser'], ['Brush', 'brush'], ['Shapes', 'shapes'], ['Fill', 'fill'], ['Text', 'text']];
  return (
    <aside className="toolbox panel" aria-label="Toolbox">
      <h2 className="panel-heading toolbox-heading">Toolbox</h2>
      <div className="main-tools">{tools.map(([label, icon]) => <button key={label} type="button" className={`main-tool${activeTool === label ? ' selected' : ''}`} onClick={() => { setActiveTool(label); if (label === 'Shapes') setShapesExpanded((expanded) => !expanded); }} aria-pressed={activeTool === label}><Icon name={icon} size={14} /><span>{label}</span>{label === 'Shapes' && <span className="tool-caret" aria-hidden="true">{shapesExpanded ? '⌄' : '›'}</span>}</button>)}</div>
      <section className="tool-group palette-group"><h3>Color Palette</h3><div className="palette-grid" role="group" aria-label="Color palette">{palette.map(([name, value]) => <button key={name} type="button" className={`color-swatch${selectedColor === value ? ' chosen' : ''}`} style={{ '--swatch': value }} aria-label={name} aria-pressed={selectedColor === value} onClick={() => setSelectedColor(value)} />)}<button className="color-wheel" type="button" disabled aria-label="Color palette swatch"><span /></button><label className="custom-color" aria-label="Choose custom color" style={{ '--swatch': selectedColor }}><input type="color" value={selectedColor} onChange={(event) => setSelectedColor(event.target.value)} aria-label="Custom color" /><Icon name="pencil" size={12} /></label></div></section>
      <section className="tool-group slider-group"><div className="slider-title"><h3>Brush Size</h3><output>{brushSize}</output></div><input aria-label="Brush size" type="range" min="1" max="20" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} /><div className="slider-scale"><span>1</span><span>20</span></div></section>
      <section className="tool-group slider-group opacity-group"><div className="slider-title"><h3>Opacity</h3><output>{opacity}%</output></div><input aria-label="Opacity" type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></section>
      <section className="tool-group shape-group"><div className="shape-heading"><h3>Shape Tools</h3><button type="button" aria-label={shapesExpanded ? 'Collapse shape tools' : 'Expand shape tools'} aria-expanded={shapesExpanded} onClick={() => setShapesExpanded((expanded) => !expanded)}>{shapesExpanded ? '⌃' : '⌄'}</button></div>{shapesExpanded && <div className="shape-grid">{shapeTools.map(([name, icon]) => <button key={name} type="button" className={`shape-tool${activeTool === name ? ' selected' : ''}`} aria-label={name} aria-pressed={activeTool === name} onClick={() => setActiveTool(name)}><Icon name={icon} size={17} /></button>)}</div>}</section>
    </aside>
  );
}

function DrawingPanel({ activeTool, brushSize, opacity, selectedColor, hasDrawing, onDrawingChange, historyStatus, onHistoryChange, onNotice, onGenerationError, onFallbackFullscreen, uploadHandlerRef, onUploadStatus, downloadHandlerRef, generationHandlerRef, onGenerateSketch }) {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const contextRef = useRef(null);
  const overlayContextRef = useRef(null);
  const canvasSizeRef = useRef(null);
  const strokeRef = useRef(null);
  const uploadTokenRef = useRef(0);
  const uploadUrlRef = useRef(null);
  const historyRef = useRef({ states: [], index: -1 });
  const drawingPresentRef = useRef(hasDrawing);
  const [zoom, setZoom] = useState(100);
  const [textDraft, setTextDraft] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const zoomScale = zoom / 100;

  const setDrawingPresence = (present) => {
    drawingPresentRef.current = present;
    onDrawingChange(present);
  };

  const captureHistoryState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const snapshot = document.createElement('canvas');
    snapshot.width = canvas.width;
    snapshot.height = canvas.height;
    snapshot.getContext('2d')?.drawImage(canvas, 0, 0);
    return { canvas: snapshot, hasDrawing: drawingPresentRef.current };
  };

  const recordHistoryState = () => {
    const snapshot = captureHistoryState();
    if (!snapshot) return;
    const history = historyRef.current;
    if (history.index < 0) {
      history.states = [snapshot];
      history.index = 0;
    } else {
      history.states = history.states.slice(0, history.index + 1);
      history.states.push(snapshot);
      if (history.states.length > 30) history.states.shift();
      history.index = history.states.length - 1;
    }
    onHistoryChange({ canUndo: history.index > 0, canRedo: false });
  };

  const restoreHistoryState = (snapshot) => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (!canvas || !context || !snapshot) return;
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = 1;
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);

    const scale = Math.min(canvas.width / snapshot.canvas.width, canvas.height / snapshot.canvas.height);
    const width = snapshot.canvas.width * scale;
    const height = snapshot.canvas.height * scale;
    context.drawImage(snapshot.canvas, 0, 0, snapshot.canvas.width, snapshot.canvas.height,
      (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
    context.restore();
    overlayContextRef.current?.clearRect(0, 0, canvasSizeRef.current?.width ?? 0, canvasSizeRef.current?.height ?? 0);
    setDrawingPresence(snapshot.hasDrawing);
  };

  const undo = () => {
    const history = historyRef.current;
    if (history.index <= 0) return;
    history.index -= 1;
    restoreHistoryState(history.states[history.index]);
    onHistoryChange({ canUndo: history.index > 0, canRedo: true });
  };

  const redo = () => {
    const history = historyRef.current;
    if (history.index >= history.states.length - 1) return;
    history.index += 1;
    restoreHistoryState(history.states[history.index]);
    onHistoryChange({ canUndo: true, canRedo: history.index < history.states.length - 1 });
  };

  const commitCurrentStroke = () => {
    const stroke = strokeRef.current;
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const overlay = overlayRef.current;
    const overlayContext = overlayContextRef.current;
    const size = canvasSizeRef.current;
    if (!stroke || !canvas || !context || !overlay || !overlayContext || !size) return;
    if (!stroke.erasing) {
      context.globalCompositeOperation = 'source-over';
      context.globalAlpha = stroke.alpha;
      context.drawImage(overlay, 0, 0, overlay.width, overlay.height, 0, 0, size.width, size.height);
      context.globalAlpha = 1;
    }
    overlayContext.clearRect(0, 0, size.width, size.height);
    strokeRef.current = null;
  };

  const finishCurrentStroke = () => {
    if (!strokeRef.current) return false;
    const stroke = strokeRef.current;
    if (stroke.shape) {
      const size = canvasSizeRef.current;
      overlayContextRef.current?.clearRect(0, 0, size?.width ?? 0, size?.height ?? 0);
      strokeRef.current = null;
      return false;
    }
    if (stroke.erasing) {
      const size = canvasSizeRef.current;
      overlayContextRef.current?.clearRect(0, 0, size?.width ?? 0, size?.height ?? 0);
      strokeRef.current = null;
    } else {
      commitCurrentStroke();
    }
    recordHistoryState();
    return true;
  };

  const drawShape = (context, shape, start, end) => {
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);
    context.beginPath();
    context.lineCap = 'round';
    context.lineJoin = 'round';

    if (shape === 'Line') {
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
    } else if (shape === 'Rectangle') {
      context.rect(left, top, width, height);
    } else if (shape === 'Circle') {
      context.ellipse(left + width / 2, top + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
    } else if (shape === 'Triangle') {
      context.moveTo(left + width / 2, top);
      context.lineTo(left + width, top + height);
      context.lineTo(left, top + height);
      context.closePath();
    } else if (shape === 'Star') {
      const centerX = left + width / 2;
      const centerY = top + height / 2;
      const outerRadius = Math.min(width, height) / 2;
      const innerRadius = outerRadius * 0.42;
      for (let point = 0; point < 10; point += 1) {
        const radius = point % 2 === 0 ? outerRadius : innerRadius;
        const angle = -Math.PI / 2 + (point * Math.PI) / 5;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        if (point === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.closePath();
    } else if (shape === 'Arrow') {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const angle = Math.atan2(dy, dx);
      const length = Math.hypot(dx, dy);
      const headLength = Math.min(Math.max(8, context.lineWidth * 3), length * 0.35);
      const wingAngle = Math.PI / 6;
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.moveTo(end.x, end.y);
      context.lineTo(end.x - headLength * Math.cos(angle - wingAngle), end.y - headLength * Math.sin(angle - wingAngle));
      context.moveTo(end.x, end.y);
      context.lineTo(end.x - headLength * Math.cos(angle + wingAngle), end.y - headLength * Math.sin(angle + wingAngle));
    }
    context.stroke();
  };

  const clearShapePreview = (stroke = strokeRef.current) => {
    const size = canvasSizeRef.current;
    overlayContextRef.current?.clearRect(0, 0, size?.width ?? 0, size?.height ?? 0);
    if (stroke && strokeRef.current === stroke) strokeRef.current = null;
  };

  const commitShape = () => {
    const stroke = strokeRef.current;
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const overlay = overlayRef.current;
    const overlayContext = overlayContextRef.current;
    const size = canvasSizeRef.current;
    if (!stroke?.shape || !canvas || !context || !overlay || !overlayContext || !size) return false;
    const distance = Math.hypot(stroke.point.x - stroke.start.x, stroke.point.y - stroke.start.y);
    if (distance < 2) {
      clearShapePreview(stroke);
      return false;
    }
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = stroke.alpha;
    context.drawImage(overlay, 0, 0, overlay.width, overlay.height, 0, 0, size.width, size.height);
    context.globalAlpha = 1;
    overlayContext.clearRect(0, 0, size.width, size.height);
    strokeRef.current = null;
    setDrawingPresence(true);
    recordHistoryState();
    return true;
  };

  const fillAt = (point) => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const size = canvasSizeRef.current;
    if (!canvas || !context || !size || opacity <= 0) return;
    const pixelCount = canvas.width * canvas.height;
    const maxPixels = 4_500_000;
    if (pixelCount > maxPixels) {
      onNotice('Fill skipped: this canvas is too large for a safe fill operation.');
      return;
    }
    const image = context.getImageData(0, 0, canvas.width, canvas.height);
    const { data } = image;
    const seedX = Math.max(0, Math.min(canvas.width - 1, Math.floor(point.x * canvas.width / size.width)));
    const seedY = Math.max(0, Math.min(canvas.height - 1, Math.floor(point.y * canvas.height / size.height)));
    const seed = (seedY * canvas.width + seedX) * 4;
    const seedR = data[seed];
    const seedG = data[seed + 1];
    const seedB = data[seed + 2];
    const color = selectedColor.match(/[0-9a-f]{2}/gi)?.map((channel) => parseInt(channel, 16));
    if (!color || color.length < 3) return;
    const visited = new Uint8Array(pixelCount);
    const queue = new Int32Array(pixelCount);
    const tolerance = 28;
    let head = 0;
    let tail = 1;
    queue[0] = seedY * canvas.width + seedX;
    visited[queue[0]] = 1;
    while (head < tail) {
      const index = queue[head++];
      const offset = index * 4;
      if (Math.abs(data[offset] - seedR) > tolerance || Math.abs(data[offset + 1] - seedG) > tolerance || Math.abs(data[offset + 2] - seedB) > tolerance) continue;
      const x = index % canvas.width;
      const add = (next) => {
        const nextOffset = next * 4;
        if (!visited[next] && Math.abs(data[nextOffset] - seedR) <= tolerance && Math.abs(data[nextOffset + 1] - seedG) <= tolerance && Math.abs(data[nextOffset + 2] - seedB) <= tolerance) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      };
      if (x > 0) add(index - 1);
      if (x < canvas.width - 1) add(index + 1);
      if (index >= canvas.width) add(index - canvas.width);
      if (index < pixelCount - canvas.width) add(index + canvas.width);
    }
    const maxRegion = pixelCount * 0.4;
    if (tail > maxRegion) {
      onNotice('Fill skipped: the area appears open or too large. Enclose it with a stroke first.');
      return;
    }
    const alpha = opacity / 100;
    for (let i = 0; i < tail; i += 1) {
      const offset = queue[i] * 4;
      data[offset] = Math.round(color[0] * alpha + data[offset] * (1 - alpha));
      data[offset + 1] = Math.round(color[1] * alpha + data[offset + 1] * (1 - alpha));
      data[offset + 2] = Math.round(color[2] * alpha + data[offset + 2] * (1 - alpha));
      data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    onNotice('');
    setDrawingPresence(true);
    recordHistoryState();
  };

  const commitText = (event) => {
    event.preventDefault();
    if (!textDraft?.text.trim()) {
      setTextDraft(null);
      return;
    }
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const size = canvasSizeRef.current;
    if (!canvas || !context || !size) return;
    context.save();
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = opacity / 100;
    context.fillStyle = selectedColor;
    context.font = `${textDraft.fontSize}px system-ui, sans-serif`;
    context.textBaseline = 'top';
    const maxWidth = Math.max(20, size.width - textDraft.point.x - 8);
    const lineHeight = textDraft.fontSize * 1.25;
    const words = textDraft.text.trim().split(/\s+/);
    const lines = [];
    let line = '';
    for (const word of words) {
      if (context.measureText(word).width > maxWidth) {
        if (line) lines.push(line);
        line = '';
        for (const character of word) {
          if (line && context.measureText(line + character).width > maxWidth) {
            lines.push(line);
            line = character;
          } else line += character;
        }
        continue;
      }
      const candidate = line ? `${line} ${word}` : word;
      if (context.measureText(candidate).width <= maxWidth) line = candidate;
      else {
        lines.push(line);
        line = word;
      }
    }
    if (line) lines.push(line);
    const maxLines = Math.max(1, Math.floor((size.height - textDraft.point.y - 4) / lineHeight));
    lines.slice(0, maxLines).forEach((textLine, index) => context.fillText(textLine, textDraft.point.x, textDraft.point.y + index * lineHeight, maxWidth));
    context.restore();
    setDrawingPresence(true);
    setTextDraft(null);
    recordHistoryState();
  };

  const uploadImage = async (file) => {
    if (!file) return;
    const token = uploadTokenRef.current + 1;
    uploadTokenRef.current = token;
    if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
    uploadUrlRef.current = null;
    const allowedMimeTypes = ['image/png', 'image/jpeg'];
    const hasSupportedExtension = /\.(png|jpe?g)$/i.test(file.name);
    const hasSupportedType = !file.type || allowedMimeTypes.includes(file.type.toLowerCase());
    if (!hasSupportedExtension || !hasSupportedType) {
      onUploadStatus('Unsupported file. Choose a PNG, JPG, or JPEG image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      onUploadStatus('File is larger than 10 MB. Choose a smaller image.');
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    uploadUrlRef.current = objectUrl;
    onUploadStatus('Loading image…');

    try {
      const image = new Image();
      image.decoding = 'async';
      if (typeof image.decode === 'function') {
        image.src = objectUrl;
        await image.decode();
      } else {
        await new Promise((resolve, reject) => {
          image.onload = resolve;
          image.onerror = () => reject(new Error('Image could not be decoded.'));
          image.src = objectUrl;
        });
      }
      if (token !== uploadTokenRef.current) return;
      if (!image.naturalWidth || !image.naturalHeight) throw new Error('Image has no readable dimensions.');

      const canvas = canvasRef.current;
      const context = contextRef.current;
      const overlay = overlayRef.current;
      const overlayContext = overlayContextRef.current;
      const size = canvasSizeRef.current;
      if (!canvas || !context || !overlay || !overlayContext || !size) throw new Error('Canvas is not ready.');
      finishCurrentStroke();
      context.save();
      context.globalCompositeOperation = 'source-over';
      context.globalAlpha = 1;
      context.shadowBlur = 0;
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, size.width, size.height);
      const scale = Math.min(size.width / image.naturalWidth, size.height / image.naturalHeight);
      const width = image.naturalWidth * scale;
      const height = image.naturalHeight * scale;
      context.drawImage(image, (size.width - width) / 2, (size.height - height) / 2, width, height);
      context.restore();
      overlayContext.clearRect(0, 0, size.width, size.height);
      setDrawingPresence(true);
      recordHistoryState();
      onUploadStatus(`Loaded ${file.name}. You can draw over the image.`);
    } catch {
      if (token === uploadTokenRef.current) onUploadStatus('This image could not be opened. Try another PNG, JPG, or JPEG file.');
    } finally {
      URL.revokeObjectURL(objectUrl);
      if (uploadUrlRef.current === objectUrl) uploadUrlRef.current = null;
    }
  };

  uploadHandlerRef.current = uploadImage;

  const toggleFullscreen = async () => {
    const shell = document.getElementById('workspace');
    if (!shell) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen?.();
      return;
    }
    if (isFullscreen) {
      onFallbackFullscreen(false);
      setIsFullscreen(false);
      return;
    }
    if (shell.requestFullscreen) {
      try {
        await shell.requestFullscreen();
        return;
      } catch {
        // Fall back to an in-page expanded workspace when native fullscreen is blocked.
      }
    }
    onFallbackFullscreen(true);
    setIsFullscreen(true);
  };

  useEffect(() => {
    const stroke = strokeRef.current;
    if (stroke?.shape && stroke.shape !== activeTool) clearShapePreview(stroke);
    if (textDraft && activeTool !== 'Text') setTextDraft(null);
  }, [activeTool]);

  useEffect(() => {
    const syncFullscreen = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', syncFullscreen);
    return () => document.removeEventListener('fullscreenchange', syncFullscreen);
  }, []);

  useEffect(() => {
    if (!isFullscreen || document.fullscreenElement) return undefined;
    const exitFallback = (event) => {
      if (event.key === 'Escape') {
        onFallbackFullscreen(false);
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', exitFallback);
    return () => window.removeEventListener('keydown', exitFallback);
  }, [isFullscreen, onFallbackFullscreen]);

  useEffect(() => {
    const handleShortcut = (event) => {
      if (event.key === 'Escape') {
        if (textDraft) setTextDraft(null);
        const stroke = strokeRef.current;
        if (stroke?.shape) {
          clearShapePreview(stroke);
          if (overlayRef.current?.hasPointerCapture(stroke.pointerId)) overlayRef.current.releasePointerCapture(stroke.pointerId);
        }
        return;
      }
      const target = event.target;
      const isTyping = target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
      if (isTyping || !(event.ctrlKey || event.metaKey)) return;
      if (event.key.toLowerCase() === 'z' && event.shiftKey) {
        event.preventDefault();
        redo();
      } else if (event.key.toLowerCase() === 'z') {
        event.preventDefault();
        undo();
      } else if (event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [textDraft]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const resizeCanvas = () => {
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;
      if (!width || !height) return;
      const ratio = window.devicePixelRatio || 1;
      const nextWidth = Math.max(1, Math.round(width * ratio));
      const nextHeight = Math.max(1, Math.round(height * ratio));
      if (canvas.width === nextWidth && canvas.height === nextHeight) return;

      const previousSize = canvasSizeRef.current;
      let snapshot = null;
      if (canvas.width && canvas.height) {
        finishCurrentStroke();
        snapshot = document.createElement('canvas');
        snapshot.width = canvas.width;
        snapshot.height = canvas.height;
        snapshot.getContext('2d')?.drawImage(canvas, 0, 0);
      }

      strokeRef.current = null;
      canvas.width = nextWidth;
      canvas.height = nextHeight;
      const overlay = overlayRef.current;
      if (!overlay) return;
      overlay.width = nextWidth;
      overlay.height = nextHeight;
      const context = canvas.getContext('2d');
      const overlayContext = overlay.getContext('2d');
      if (!context || !overlayContext) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, width, height);

      if (snapshot && previousSize) {
        // Fit the previous drawing inside the new canvas without changing its proportions.
        const scale = Math.min(width / previousSize.width, height / previousSize.height);
        const drawWidth = previousSize.width * scale;
        const drawHeight = previousSize.height * scale;
        context.drawImage(snapshot, 0, 0, snapshot.width, snapshot.height,
          (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
      }
      context.lineCap = 'round';
      context.lineJoin = 'round';
      contextRef.current = context;
      overlayContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      overlayContext.lineCap = 'round';
      overlayContext.lineJoin = 'round';
      overlayContextRef.current = overlayContext;
      canvasSizeRef.current = { width: canvas.offsetWidth, height: canvas.offsetHeight };
      if (historyRef.current.index < 0) recordHistoryState();
    };

    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(canvas);
    resizeCanvas();
    const onWindowResize = () => resizeCanvas();
    window.addEventListener('resize', onWindowResize);
    const onWindowBlur = () => {
      const stroke = strokeRef.current;
      finishCurrentStroke();
      if (stroke && overlayRef.current?.hasPointerCapture(stroke.pointerId)) overlayRef.current.releasePointerCapture(stroke.pointerId);
    };
    window.addEventListener('blur', onWindowBlur);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', onWindowResize);
      window.removeEventListener('blur', onWindowBlur);
      uploadTokenRef.current += 1;
      if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
      uploadUrlRef.current = null;
    };
  }, []);

  const getPoint = (event) => {
    const canvas = overlayRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.clientWidth, (event.clientX - rect.left) / zoomScale)),
      y: Math.max(0, Math.min(canvas.clientHeight, (event.clientY - rect.top) / zoomScale)),
    };
  };

  const handlePointerDown = (event) => {
    const shapeTool = ['Line', 'Rectangle', 'Circle', 'Triangle', 'Star', 'Arrow'].includes(activeTool);
    if (activeTool === 'Text') {
      event.preventDefault();
      setTextDraft({ point: getPoint(event), text: '', fontSize: Math.max(12, brushSize * 2) });
      return;
    }
    if (activeTool === 'Fill') {
      event.preventDefault();
      fillAt(getPoint(event));
      return;
    }
    const brushTool = activeTool === 'Brush';
    if (activeTool !== 'Pencil' && activeTool !== 'Eraser' && !brushTool && !shapeTool) return;
    event.preventDefault();
    if (strokeRef.current) finishCurrentStroke();
    const erasing = activeTool === 'Eraser';
    const context = erasing ? contextRef.current : overlayContextRef.current;
    const canvas = overlayRef.current;
    if (!context || !canvas) return;
    const point = getPoint(event);
    if (!erasing) overlayContextRef.current.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = 1;
    context.strokeStyle = erasing ? '#ffffff' : selectedColor;
    context.fillStyle = erasing ? '#ffffff' : selectedColor;
    context.lineWidth = brushSize;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.shadowColor = brushTool ? selectedColor : 'transparent';
    context.shadowBlur = brushTool ? Math.max(1, brushSize * 0.55) : 0;
    if (shapeTool) {
      strokeRef.current = { pointerId: event.pointerId, point, start: point, alpha: opacity / 100, shape: activeTool, context };
    } else {
      context.beginPath();
      context.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
      context.fill();
      strokeRef.current = { pointerId: event.pointerId, point, alpha: erasing ? 1 : opacity / 100, erasing, brush: brushTool, context };
      setDrawingPresence(true);
    }
    canvas.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event) => {
    const stroke = strokeRef.current;
    if (!stroke || stroke.pointerId !== event.pointerId || !stroke.context) return;
    event.preventDefault();
    if (stroke.shape) {
      const point = getPoint(event);
      stroke.point = point;
      const previewContext = overlayContextRef.current;
      const size = canvasSizeRef.current;
      if (!previewContext || !size) return;
      previewContext.clearRect(0, 0, size.width, size.height);
      previewContext.globalCompositeOperation = 'source-over';
      previewContext.globalAlpha = 1;
      previewContext.strokeStyle = selectedColor;
      previewContext.lineWidth = brushSize;
      drawShape(previewContext, stroke.shape, stroke.start, point);
      return;
    }
    const context = stroke.context;
    const samples = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [event];
    for (const sample of (samples.length ? samples : [event])) {
      const point = getPoint(sample);
      context.beginPath();
      context.moveTo(stroke.point.x, stroke.point.y);
      context.lineTo(point.x, point.y);
      context.stroke();
      stroke.point = point;
    }
  };

  const finishPointer = (event) => {
    if (strokeRef.current?.pointerId !== event.pointerId) return;
    if (strokeRef.current.shape) commitShape();
    else finishCurrentStroke();
    if (overlayRef.current?.hasPointerCapture(event.pointerId)) overlayRef.current.releasePointerCapture(event.pointerId);
  };

  const cancelPointer = (event) => {
    if (strokeRef.current?.pointerId !== event.pointerId) return;
    if (strokeRef.current.shape) clearShapePreview(strokeRef.current);
    else finishCurrentStroke();
    if (overlayRef.current?.hasPointerCapture(event.pointerId)) overlayRef.current.releasePointerCapture(event.pointerId);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const overlay = overlayRef.current;
    const overlayContext = overlayContextRef.current;
    if (!canvas || !context || !overlay || !overlayContext) return;
    uploadTokenRef.current += 1;
    if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
    uploadUrlRef.current = null;
    strokeRef.current = null;
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = 1;
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    overlayContext.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
    setDrawingPresence(false);
    onUploadStatus('');
    recordHistoryState();
  };

  const downloadSketch = () => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    if (!canvas || !overlay) {
      onNotice('Could not export the sketch. The canvas is not ready.');
      return;
    }
    try {
      const exportCanvas = document.createElement('canvas');
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const exportContext = exportCanvas.getContext('2d');
      if (!exportContext) throw new Error('Canvas export is unavailable.');
      exportContext.fillStyle = '#ffffff';
      exportContext.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      exportContext.drawImage(canvas, 0, 0);
      const translucentTools = ['Pencil', 'Brush', 'Line', 'Rectangle', 'Circle', 'Triangle', 'Star', 'Arrow'];
      exportContext.globalAlpha = translucentTools.includes(activeTool) ? opacity / 100 : 1;
      exportContext.drawImage(overlay, 0, 0);
      exportContext.globalAlpha = 1;
      exportCanvas.toBlob((blob) => {
        if (!blob) {
          onNotice('Could not create a PNG from this sketch.');
          return;
        }
        try {
          const objectUrl = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = objectUrl;
          link.download = 'sketch2vision-sketch.png';
          link.style.display = 'none';
          document.body.appendChild(link);
          link.click();
          link.remove();
          window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
          onNotice('Sketch downloaded as PNG.');
        } catch {
          onNotice('Could not start the sketch download.');
        }
      }, 'image/png');
    } catch {
      onNotice('Could not export the sketch as PNG.');
    }
  };

  downloadHandlerRef.current = downloadSketch;

  const prepareSketchForDescription = async () => {
    try {
      const canvas = canvasRef.current;
      const overlay = overlayRef.current;
      if (!hasDrawing) throw new Error('Draw or upload a sketch before generating an image.');
      if (!canvas || !overlay) throw new Error('The drawing canvas is not ready.');
      const exportCanvas = document.createElement('canvas');
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const exportContext = exportCanvas.getContext('2d');
      if (!exportContext) throw new Error('Could not prepare the sketch image.');
      exportContext.fillStyle = '#ffffff';
      exportContext.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      exportContext.drawImage(canvas, 0, 0);
      const translucentTools = ['Pencil', 'Brush', 'Line', 'Rectangle', 'Circle', 'Triangle', 'Star', 'Arrow'];
      exportContext.globalAlpha = translucentTools.includes(activeTool) ? opacity / 100 : 1;
      exportContext.drawImage(overlay, 0, 0);
      exportContext.globalAlpha = 1;
      const blob = await new Promise((resolve, reject) => exportCanvas.toBlob((value) => value ? resolve(value) : reject(new Error('Could not encode the sketch as PNG.')), 'image/png'));
      await onGenerateSketch(blob);
    } catch (error) {
      onGenerationError(error.message || 'Could not prepare the sketch for description.');
    }
  };

  generationHandlerRef.current = prepareSketchForDescription;

  const textEntryPosition = () => {
    const size = canvasSizeRef.current;
    if (!size || !textDraft) return { left: 8, top: 8 };
    const x = size.width / 2 + (textDraft.point.x - size.width / 2) * zoomScale;
    const y = size.height / 2 + (textDraft.point.y - size.height / 2) * zoomScale;
    return { left: Math.max(6, Math.min(x, size.width - 236)), top: Math.max(6, Math.min(y, size.height - 128)) };
  };

  return (
      <section className="drawing-panel panel" aria-labelledby="drawing-title">
      <div className="panel-title-row"><h2 id="drawing-title">Drawing Canvas</h2><div className="canvas-actions"><button type="button" disabled={!historyStatus.canUndo} onClick={undo}><Icon name="undo" size={13} />Undo</button><button type="button" disabled={!historyStatus.canRedo} onClick={redo}><Icon name="redo" size={13} />Redo</button><button type="button" className="clear-action" disabled={!hasDrawing} onClick={() => { if (window.confirm('Clear the current drawing?')) clearCanvas(); }}><Icon name="trash" size={13} />Clear</button></div></div>
      <div className="canvas-stack">
        <div className="canvas-zoom-surface" style={{ transform: `scale(${zoomScale})` }}>
          <canvas ref={canvasRef} className="white-canvas" role="img" aria-label="Interactive white drawing canvas" />
          <canvas ref={overlayRef} className={`stroke-overlay tool-${activeTool.toLowerCase()}`} aria-hidden="true" style={{ opacity: ['Pencil', 'Brush', 'Line', 'Rectangle', 'Circle', 'Triangle', 'Star', 'Arrow'].includes(activeTool) ? opacity / 100 : 1 }} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={finishPointer} onPointerCancel={cancelPointer} />
        </div>
        {textDraft && <form className="text-entry" style={textEntryPosition()} onSubmit={commitText} onPointerDown={(event) => event.stopPropagation()}>
          <label htmlFor="canvas-text-input">Text to place</label>
          <input id="canvas-text-input" autoFocus value={textDraft.text} onChange={(event) => setTextDraft({ ...textDraft, text: event.target.value })} maxLength={500} />
          <label htmlFor="canvas-text-size">Font size</label>
          <input id="canvas-text-size" className="text-size-input" type="number" min="8" max="72" value={textDraft.fontSize} onChange={(event) => setTextDraft({ ...textDraft, fontSize: Math.max(8, Math.min(72, Number(event.target.value) || 8)) })} />
          <div><button type="submit">Place text</button><button type="button" onClick={() => setTextDraft(null)}>Cancel</button></div>
        </form>}
      </div>
      <div className="canvas-controls"><label className="zoom-label" htmlFor="canvas-zoom">Zoom:</label><input id="canvas-zoom" type="range" min="25" max="200" step="5" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} aria-label="Canvas zoom" /><span className="zoom-percent" aria-live="polite">{zoom}%</span><button className="fit-control" type="button" onClick={() => setZoom(100)} title="Fit canvas">Fit</button><button className="fullscreen-control" type="button" onClick={toggleFullscreen} aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}><span /></button></div>
    </section>
  );
}

function DescriptionPanel({ onDownloadSketch, onDescribe, canDescribe, isDescribing, description, error, elapsedSeconds }) {
  return (
    <section className="result-panel panel description-panel" aria-labelledby="description-title">
      <div className="panel-title-row result-title-row"><h2 id="description-title">Sketch Description</h2><span className="time-badge"><Icon name="clock" size={12} />{elapsedSeconds == null ? 'Ready' : `${elapsedSeconds.toFixed(1)} s`}</span></div>
      <div className={`description-output${isDescribing ? ' is-generating' : ''}`} aria-live="polite" aria-busy={isDescribing}>
        {isDescribing ? <><span className="large-spinner" aria-hidden="true" /><p>Analyzing your sketch…</p></> : description ? <p className="description-text">{description}</p> : <><Icon name="image" size={24} /><p>Your sketch description will appear here.</p><small>Draw on the canvas, then choose Describe Sketch.</small></>}
        {error && <p className="generation-error" role="alert">{error}</p>}
      </div>
      <div className="result-actions description-actions"><button className="result-generate" type="button" disabled={!canDescribe || isDescribing} onClick={onDescribe}>{isDescribing ? <span className="button-spinner" aria-hidden="true" /> : <Icon name="sparkle" size={13} />}{isDescribing ? 'Describing…' : description ? 'Describe Again' : 'Describe Sketch'}</button><button type="button" onClick={onDownloadSketch} aria-label="Download sketch as PNG" title="Download sketch as PNG"><Icon name="download" size={13} />Download Sketch</button></div>
      <p className="model-note">Description is produced by Groq’s vision-capable Qwen model using your submitted sketch. Results may be uncertain for abstract or incomplete drawings.</p>
    </section>
  );
}

function UploadPanel({ inputRef, onUpload, onOpenPicker, status }) {
  const [dragging, setDragging] = useState(false);
  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    onUpload(event.dataTransfer?.files?.[0]);
  };
  return <section className="upload-panel panel" aria-labelledby="upload-title">
    <h2 id="upload-title">Upload a Sketch</h2>
    <input ref={inputRef} className="file-input-hidden" type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" aria-label="Choose a PNG or JPEG sketch" onChange={(event) => { onUpload(event.target.files?.[0]); event.target.value = ''; }} />
    <button className={`drop-zone${dragging ? ' dragging' : ''}`} type="button" onClick={onOpenPicker} onDragEnter={(event) => { event.preventDefault(); setDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }} onDrop={handleDrop} aria-label="Choose or drop a sketch image">
      <Icon name="upload" size={24} /><strong>Select or drop an image here</strong><span>PNG, JPG, or JPEG · Up to 10 MB</span><small>Your image stays in this browser.</small>{status && <output className="upload-status" aria-live="polite">{status}</output>}
    </button>
  </section>;
}

function ExamplesPanel() {
  return <section className="examples-panel panel" aria-labelledby="examples-title"><h2 id="examples-title">Example Sketches</h2><div className="example-tiles">{['House', 'Car', 'Cat', 'Chair', 'Bag'].map((label, index) => <div className="example-tile" key={label} aria-label={`${label} placeholder`}><span>{String(index + 1).padStart(2, '0')}</span><Icon name="image" size={20} /></div>)}</div></section>;
}

export default function App() {
  const [activeTool, setActiveTool] = useState('Pencil');
  const [brushSize, setBrushSize] = useState(5);
  const [opacity, setOpacity] = useState(100);
  const [selectedColor, setSelectedColor] = useState('#252525');
  const [hasDrawing, setHasDrawing] = useState(false);
  const [historyStatus, setHistoryStatus] = useState({ canUndo: false, canRedo: false });
  const [shapesExpanded, setShapesExpanded] = useState(true);
  const [notice, setNotice] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [description, setDescription] = useState('');
  const [descriptionError, setDescriptionError] = useState('');
  const [descriptionSeconds, setDescriptionSeconds] = useState(null);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const uploadInputRef = useRef(null);
  const uploadHandlerRef = useRef(null);
  const downloadHandlerRef = useRef(null);
  const generationHandlerRef = useRef(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const handleUploadFile = (file) => uploadHandlerRef.current?.(file);
  const openUploadPicker = () => uploadInputRef.current?.click();
  const describeSketch = async (blob) => {
    if (isGenerating) return;
    setIsGenerating(true);
    setDescriptionError('');
    setNotice('Sending your sketch to Groq for description…');
    const started = performance.now();
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'}/api/describe`, {
        method: 'POST', headers: { 'Content-Type': 'image/png' }, body: blob,
      });
      if (!response.ok) {
        let detail = `Description failed (HTTP ${response.status}).`;
        try { const payload = await response.json(); if (payload?.detail) detail = payload.detail; } catch { /* retain HTTP message */ }
        throw new Error(detail);
      }
      const payload = await response.json();
      if (typeof payload?.description !== 'string' || !payload.description.trim()) throw new Error('The backend returned an empty description.');
      setDescription(payload.description.trim());
      setDescriptionSeconds((performance.now() - started) / 1000);
      setNotice('Sketch description received.');
    } catch (error) {
      const detail = error instanceof TypeError ? 'Could not reach the backend. Start the FastAPI server and try again.' : error.message || 'Sketch description failed.';
      setDescriptionError(detail);
      setNotice('Sketch description failed.');
    } finally {
      setIsGenerating(false);
    }
  };
  const triggerDescribe = () => generationHandlerRef.current?.();
  return <div className={`app-shell${fallbackFullscreen ? ' fallback-fullscreen' : ''}`} id="workspace"><Header onUploadClick={openUploadPicker} onDescribe={triggerDescribe} isDescribing={isGenerating} canDescribe={hasDrawing} /><main className="workspace-grid"><Toolbox activeTool={activeTool} setActiveTool={setActiveTool} brushSize={brushSize} setBrushSize={setBrushSize} opacity={opacity} setOpacity={setOpacity} selectedColor={selectedColor} setSelectedColor={setSelectedColor} shapesExpanded={shapesExpanded} setShapesExpanded={setShapesExpanded} /><DrawingPanel activeTool={activeTool} brushSize={brushSize} opacity={opacity} selectedColor={selectedColor} hasDrawing={hasDrawing} onDrawingChange={setHasDrawing} historyStatus={historyStatus} onHistoryChange={setHistoryStatus} onNotice={setNotice} onGenerationError={(message) => { setDescriptionError(message); setNotice('Could not prepare the sketch.'); }} onFallbackFullscreen={setFallbackFullscreen} uploadHandlerRef={uploadHandlerRef} onUploadStatus={setUploadStatus} downloadHandlerRef={downloadHandlerRef} generationHandlerRef={generationHandlerRef} onGenerateSketch={describeSketch} /><DescriptionPanel onDownloadSketch={() => downloadHandlerRef.current?.()} onDescribe={triggerDescribe} canDescribe={hasDrawing} isDescribing={isGenerating} description={description} error={descriptionError} elapsedSeconds={descriptionSeconds} /><UploadPanel inputRef={uploadInputRef} onUpload={handleUploadFile} onOpenPicker={openUploadPicker} status={uploadStatus} /><ExamplesPanel /></main><footer className="status-footer"><span className="ready-dot" />{notice || 'Ready to create'}<span className="footer-divider" />{activeTool} selected</footer></div>;
}
