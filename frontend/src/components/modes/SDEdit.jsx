import { useState, useEffect } from 'react';
import { X, ArrowUUpLeft, Trash, Eye, Eyedropper, PaintBrush, Image as ImageIcon } from '@phosphor-icons/react';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';
import AdvancedSettings from '../AdvancedSettings';

const initialState = {
  prompt: '',
  strength: 0.6,
  color: null,
  strokes: {},
  numInferenceSteps: '',
  seed: '',
};

// Strokes are kept under the image they were painted on (its name and size), so another image
// simply has none: nothing has to clear them when the image changes, is sent in or is restored.
const keyOf = (images) => images[0] && `${images[0].name}:${images[0].size}`;

// The pipette: the photo's pixel under a point (normalized), read off a 1px canvas.
function pick(img, [x, y]) {
  const ctx = Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d');
  ctx.drawImage(img, Math.floor(x * img.naturalWidth), Math.floor(y * img.naturalHeight), 1, 1, 0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

// What is sent: the photo with its strokes painted on, as the canvas shows them. Composed here
// rather than on the server, so the colours are the ones the pipette read off this same decode.
async function painted(file, strokes) {
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode().finally(() => URL.revokeObjectURL(img.src));
  const canvas = Object.assign(document.createElement('canvas'), { width: img.naturalWidth, height: img.naturalHeight });
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  ctx.lineCap = ctx.lineJoin = 'round';
  for (const { color, r, pts } of strokes) {
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = 2 * r * canvas.width;
    ctx.beginPath();
    for (const [x, y] of pts) ctx.lineTo(x * canvas.width, y * canvas.height);
    ctx.stroke();
    // A click is a stroke of no length: the <polyline> shows its round caps, a canvas draws nothing.
    ctx.beginPath();
    ctx.arc(pts[0][0] * canvas.width, pts[0][1] * canvas.height, r * canvas.width, 0, 2 * Math.PI);
    ctx.fill();
  }
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.95));
}

function Inputs({ state, setState, images, setImages, onZoom, onResult }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      {/* The result was made from the image this one replaces: without it the canvas shows again. */}
      <ImageDropZone images={images} onChange={(next) => { setImages(next); onResult(null); }} onZoom={onZoom} />

      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.prompt')}</span>
        <div className="relative">
          <textarea
            value={state.prompt}
            onChange={(e) => set({ prompt: e.target.value })}
            rows={3}
            placeholder={t('sdedit.prompt_example')}
            className="input-textarea"
          />
          {state.prompt && (
            <button type="button" onClick={() => set({ prompt: '' })} className="input-clear-btn">
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* SDEdit's t0: how far the painted image is noised before it is denoised again. */}
      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sdedit.strength')}</span>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="0.05"
            max="1"
            step="0.01"
            value={state.strength}
            onChange={(e) => set({ strength: parseFloat(e.target.value) })}
            className="flex-1 accent-[#0ea0ff]"
          />
          <span className="w-9 text-right text-xs tabular-nums text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{Math.round(state.strength * 100)}%</span>
        </div>
      </div>

      <AdvancedSettings>
      <div className="flex flex-col gap-4">
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.inference_steps')}</span>
          <input
            type="number"
            placeholder="50"
            value={state.numInferenceSteps}
            onChange={(e) => set({ numInferenceSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.seed')}</span>
          <input
            type="number"
            value={state.seed}
            onChange={(e) => set({ seed: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
      </div>
      </AdvancedSettings>
    </div>
  );
}

// The result card is the workspace, as in SAM3: the photo at full size, painted on with a
// colour the pipette picked from it. A run's result takes its place until a tool is picked up.
function Result({ result, state, setState, images, onZoom }) {
  const { t } = useLang();
  const [imageUrl, setImageUrl] = useState(null);
  const [dims, setDims] = useState(null);
  const [picking, setPicking] = useState(false);
  // Wider than SAM3's eraser: a thin stroke doesn't survive the noise.
  const [brush, setBrush] = useState(20);
  const [stroke, setStroke] = useState(null);
  const [hover, setHover] = useState(null);
  const [plain, setPlain] = useState(false);

  useEffect(() => {
    setDims(null);
    if (!images.length) { setImageUrl(null); return; }
    const url = URL.createObjectURL(images[0]);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [images]);

  useEffect(() => setPlain(!!result), [result]);

  const key = keyOf(images);
  const strokes = state.strokes[key] || [];
  // Until a colour has been picked there is nothing to paint with.
  const pipette = picking || !state.color;

  const press = (e) => {
    if (e.button !== 0) return;
    const frame = e.currentTarget;
    const at = (m) => {
      const b = frame.getBoundingClientRect();
      return [(m.clientX - b.left) / b.width, (m.clientY - b.top) / b.height];
    };
    if (pipette) {
      const picked = pick(frame.firstElementChild, at(e));
      setState((s) => ({ ...s, color: picked }));
      setPicking(false);
      return;
    }
    // Normalized like SAM3's strokes, the radius a fraction of the width.
    const color = state.color;
    const r = brush / frame.getBoundingClientRect().width;
    const pts = [at(e), at(e)];
    setStroke({ color, r, pts: [...pts] });
    const move = (m) => {
      if (!(m.buttons & 1)) return up();
      pts.push(at(m));
      setStroke({ color, r, pts: [...pts] });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setStroke(null);
      setState((s) => ({ ...s, strokes: { [key]: [...(s.strokes[key] || []), { color, r, pts }] } }));
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const undo = () => { setPlain(false); setState((s) => ({ ...s, strokes: { [key]: (s.strokes[key] || []).slice(0, -1) } })); };
  const clear = () => { setPlain(false); setState((s) => ({ ...s, strokes: {} })); };

  if (!imageUrl) {
    return (
      <div className="flex flex-col items-center gap-2 text-gray-300 dark:text-zinc-600">
        <ImageIcon size={40} />
      </div>
    );
  }

  const btn = 'inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 disabled:opacity-40 transition-[transform,background-color] active:scale-[0.96]';
  const on = 'bg-gray-100 dark:bg-zinc-800';

  return (
    <div className="w-full h-full min-h-0 flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-shrink-0">
        <button type="button" onClick={() => { setPicking(true); setPlain(false); }} className={`${btn} ${pipette ? on : ''}`}>
          <Eyedropper size={13} />
          {t('sdedit.pipette')}
        </button>
        <button type="button" onClick={() => { setPicking(false); setPlain(false); }} disabled={!state.color} className={`${btn} ${pipette ? '' : on}`}>
          <PaintBrush size={13} />
          {t('sdedit.brush')}
          {state.color && <span className="w-3 h-3 rounded-full border border-gray-300 dark:border-zinc-600" style={{ background: state.color }} />}
        </button>
        <button type="button" onClick={undo} disabled={!strokes.length} className={btn}>
          <ArrowUUpLeft size={13} />
          {t('sam3.undo')}
        </button>
        <button type="button" onClick={clear} disabled={!strokes.length} className={btn}>
          <Trash size={13} />
          {t('sam3.clear')}
        </button>
        <button type="button" onClick={() => setPlain(!plain)} disabled={!result} className={`${btn} ${plain && result ? on : ''}`}>
          <Eye size={13} />
          {t('common.result')}
        </button>
        {!pipette && (
          <div className="group flex items-center gap-3 ml-2">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.brush')}</span>
            <input
              type="range"
              min="1"
              max="80"
              step="1"
              value={brush}
              onChange={(e) => setBrush(parseInt(e.target.value))}
              className="w-40 accent-[#0ea0ff]"
            />
            <span className="w-9 text-xs tabular-nums text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{2 * brush}px</span>
          </div>
        )}
      </div>

      {/* A size container, so the frame below can be sized to the image fitted inside it. */}
      <div className="flex-1 min-h-0 flex items-center justify-center" style={{ containerType: 'size' }}>
        {plain && result ? (
          <img src={result} alt="Generated" onClick={() => onZoom(result)} className="max-w-full max-h-full object-contain rounded cursor-zoom-in" />
        ) : (
          // The frame hugs the image exactly — presses are read against its box. Until the
          // image has loaded and told its ratio, it is laid out unseen.
          <div
            onPointerDown={press}
            onPointerMove={(e) => {
              const b = e.currentTarget.getBoundingClientRect();
              setHover([e.clientX - b.left, e.clientY - b.top]);
            }}
            onPointerLeave={() => setHover(null)}
            style={dims ? { width: `min(100cqw, calc(100cqh * ${dims[0] / dims[1]}))` } : { width: '100%', visibility: 'hidden' }}
            className="relative rounded overflow-hidden select-none cursor-crosshair"
          >
            <img
              src={imageUrl}
              alt="Error loading image"
              draggable={false}
              onLoad={(e) => setDims([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
              className="block w-full"
            />
            {dims && (
              <svg viewBox={`0 0 ${dims[0]} ${dims[1]}`} className="absolute inset-0 w-full h-full pointer-events-none">
                {[...strokes, ...(stroke ? [stroke] : [])].map((st, i) => (
                  <polyline
                    key={i}
                    points={st.pts.map(([x, y]) => `${x * dims[0]},${y * dims[1]}`).join(' ')}
                    fill="none"
                    stroke={st.color}
                    strokeWidth={2 * st.r * dims[0]}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                ))}
              </svg>
            )}
            {!pipette && hover && (
              <span
                style={{ left: hover[0], top: hover[1], width: 2 * brush, height: 2 * brush }}
                className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.4)] pointer-events-none"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

async function submit({ images, state, run }) {
  const fd = new FormData();
  fd.append('image', await painted(images[0], state.strokes[keyOf(images)] || []));
  fd.append('prompt', state.prompt);
  fd.append('strength', state.strength);
  if (Number.isInteger(state.numInferenceSteps)) fd.append('num_inference_steps', state.numInferenceSteps);
  if (Number.isInteger(state.seed)) fd.append('seed', state.seed);
  const r = await run('/sdedit', fd);
  return { blob: await r.blob(), state };
}

export default {
  label: 'mode.sdedit',
  tasks: ['edit'],
  // FLUX.2 klein base, on the same pipeline as its own mode: SDEdit only starts it part-way.
  model: 'flux2klein',
  initialState,
  Inputs,
  Result,
  submit,
  canSubmit: ({ images }) => images.length > 0,
};
