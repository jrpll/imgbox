import { useState, useEffect } from 'react';
import { X, ArrowUUpLeft, Trash, Eraser } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';

const initialState = { points: [], labels: [], preserve: false, additive: false, erase: [], text: '' };

function Inputs({ state, setState, images, setImages, onZoom, onResult, loaded }) {
  const { t } = useLang();
  const [imageUrl, setImageUrl] = useState(null);
  const [cutout, setCutout] = useState(null);
  const [view, setView] = useState(null);
  const [zoom, setZoom] = useState({ s: 1, x: 0, y: 0 });
  const [dims, setDims] = useState(null);
  const [erasing, setErasing] = useState(false);
  const [brush, setBrush] = useState(8);
  const [stroke, setStroke] = useState(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    setZoom({ s: 1, x: 0, y: 0 });
    if (!images.length) { setImageUrl(null); return; }
    const url = URL.createObjectURL(images[0]);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [images]);

  // Scroll zooms around the cursor and left-drag pans, for precise points. Native listeners,
  // since React's wheel handlers are passive and can't keep the panel from scrolling along.
  // With the eraser on, a left press paints a stroke instead and the middle button pans.
  useEffect(() => {
    if (!view) return;
    // Keeps the zoomed layer covering the frame.
    const clamp = (v, size, s) => Math.min(0, Math.max(size * (1 - s), v));
    const onWheel = (e) => {
      e.preventDefault();
      const box = view.getBoundingClientRect();
      const cx = e.clientX - box.left, cy = e.clientY - box.top;
      const f = Math.exp(-e.deltaY * (e.deltaMode ? 40 : 1) * 0.002);
      // The pixel under the cursor stays put.
      setZoom(({ s, x, y }) => {
        const s2 = Math.min(8, Math.max(1, s * f));
        return { s: s2, x: clamp(cx - (cx - x) * s2 / s, box.width, s2), y: clamp(cy - (cy - y) * s2 / s, box.height, s2) };
      });
    };
    // A press that moves under 3px stays a click, i.e. a point; past that it pans, and the
    // click it ends with is swallowed before it reaches addPoint.
    let dragged = false;
    const onDown = (e) => {
      if (e.target.closest('button')) return;
      if (e.button === 0 && erasing) return paint(e);
      if (e.button !== 0 && e.button !== 1) return;
      const held = e.button ? 4 : 1;
      dragged = false;
      let px = e.clientX, py = e.clientY;
      const move = (m) => {
        if (!(m.buttons & held)) return up();
        const dx = m.clientX - px, dy = m.clientY - py;
        if (!dragged && Math.hypot(dx, dy) < 3) return;
        dragged = true;
        px = m.clientX; py = m.clientY;
        const box = view.getBoundingClientRect();
        setZoom(({ s, x, y }) => ({ s, x: clamp(x + dx, box.width, s), y: clamp(y + dy, box.height, s) }));
      };
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
    // Strokes are normalized like the points, with the radius a fraction of the width, so a
    // brush of fixed screen size erases finer the further you zoom in.
    const paint = (e) => {
      dragged = true;
      const layer = view.firstElementChild;
      const at = (m) => {
        const b = layer.getBoundingClientRect();
        return [(m.clientX - b.left) / b.width, (m.clientY - b.top) / b.height];
      };
      const r = brush / layer.getBoundingClientRect().width;
      const pts = [at(e), at(e)];
      setStroke({ r, pts: [...pts] });
      const move = (m) => {
        if (!(m.buttons & 1)) return up();
        pts.push(at(m));
        setStroke({ r, pts: [...pts] });
      };
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        setStroke(null);
        setState((s) => ({ ...s, erase: [...s.erase, { r, pts }] }));
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
    const onClick = (e) => { if (dragged) e.stopPropagation(); };
    const onHover = (e) => {
      const b = view.getBoundingClientRect();
      setHover([e.clientX - b.left, e.clientY - b.top]);
    };
    const onLeave = () => setHover(null);
    view.addEventListener('wheel', onWheel, { passive: false });
    view.addEventListener('pointerdown', onDown);
    view.addEventListener('click', onClick, true);
    view.addEventListener('pointermove', onHover);
    view.addEventListener('pointerleave', onLeave);
    return () => {
      view.removeEventListener('wheel', onWheel);
      view.removeEventListener('pointerdown', onDown);
      view.removeEventListener('click', onClick, true);
      view.removeEventListener('pointermove', onHover);
      view.removeEventListener('pointerleave', onLeave);
    };
  }, [view, erasing, brush]);

  // Clicking needs no Run step: every point re-cuts the image and pushes the result up.
  // The server keeps the image embedding cached, so only the mask decoder runs again.
  useEffect(() => {
    if (!images.length || !state.points.length) { setCutout(null); return; }
    let cancelled = false;
    let url = null;
    const fd = new FormData();
    fd.append('image', images[0]);
    fd.append('points', JSON.stringify(state.points));
    fd.append('labels', JSON.stringify(state.labels));
    fd.append('preserve', state.preserve);
    fd.append('additive', state.additive);
    fd.append('erase', JSON.stringify(state.erase));
    apiPost('/sam3', fd)
      .then((r) => r.blob())
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setCutout(url);
        onResult(blob);
      })
      .catch((err) => console.error('sam3:', err));
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [images, state.points, state.labels, state.preserve, state.additive, state.erase]);

  // Normalized coords so the points survive a resize and the persisted state stays
  // independent of the image resolution.
  const addPoint = (e, label) => {
    e.preventDefault();
    if (!loaded || erasing) return;
    const box = e.currentTarget.getBoundingClientRect();
    setState((s) => ({
      ...s,
      points: [...s.points, [(e.clientX - box.left) / box.width, (e.clientY - box.top) / box.height]],
      labels: [...s.labels, label],
    }));
  };

  // Undo takes back the last stroke while the eraser is on, the last point otherwise.
  // Dropping the last point leaves nothing to show, so retract the live result too.
  const undo = () => setState((s) => {
    if (erasing) return { ...s, erase: s.erase.slice(0, -1) };
    if (s.points.length === 1) onResult(null);
    return { ...s, points: s.points.slice(0, -1), labels: s.labels.slice(0, -1) };
  });
  const clear = () => { onResult(null); setState((s) => ({ ...s, points: [], labels: [], erase: [] })); };

  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  const btn = 'flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-zinc-800 disabled:opacity-40 transition-[transform,background-color] active:scale-[0.96]';

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      {!imageUrl ? (
        <ImageDropZone images={images} onChange={setImages} onZoom={onZoom} />
      ) : (
        <>
          {/* The cutout stacked over a blue wash: it is opaque wherever it was kept, so
              the wash only shows through the part that gets removed. */}
          <div className="flex flex-col gap-2">
            <div ref={setView} className={`relative rounded overflow-hidden select-none ${loaded ? 'cursor-crosshair' : ''}`}>
              {/* Clicks land on the zoomed layer, whose transformed box still maps to [0, 1]. */}
              <div
                onClick={(e) => addPoint(e, 1)}
                onContextMenu={(e) => addPoint(e, 0)}
                style={{ transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.s})`, transformOrigin: '0 0' }}
                className="relative"
              >
                <img
                  src={imageUrl}
                  alt="Error loading image"
                  draggable={false}
                  onLoad={(e) => setDims([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
                  className="block w-full"
                />
                {cutout && dims && (
                  <>
                    <span className="absolute inset-0 bg-[#0ea0ff] opacity-50 pointer-events-none" />
                    {/* Strokes are masked out of the cutout here as you paint; the server bakes
                        them into the result, so this is a no-op once its response lands. */}
                    <svg viewBox={`0 0 ${dims[0]} ${dims[1]}`} className="absolute inset-0 w-full h-full pointer-events-none">
                      <mask id="sam3-erase">
                        <rect width={dims[0]} height={dims[1]} fill="white" />
                        {[...state.erase, ...(stroke ? [stroke] : [])].map((st, i) => (
                          <polyline
                            key={i}
                            points={st.pts.map(([x, y]) => `${x * dims[0]},${y * dims[1]}`).join(' ')}
                            fill="none"
                            stroke="black"
                            strokeWidth={2 * st.r * dims[0]}
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        ))}
                      </mask>
                      <image href={cutout} width={dims[0]} height={dims[1]} mask="url(#sam3-erase)" />
                    </svg>
                  </>
                )}
                {/* Counter-scaled so dots stay the same size on screen at any zoom. */}
                {state.points.map(([x, y], i) => (
                  <span
                    key={i}
                    style={{ left: `${x * 100}%`, top: `${y * 100}%`, transform: `translate(-50%, -50%) scale(${1 / zoom.s})` }}
                    className={`absolute w-2.5 h-2.5 rounded-full border-2 border-white pointer-events-none ${
                      state.labels[i] ? 'bg-[#0ea0ff]' : 'bg-red-500'
                    }`}
                  />
                ))}
              </div>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setImages([]); clear(); }}
                className="absolute top-1 right-1 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
              >
                <X size={12} />
              </button>
              {erasing && hover && (
                <span
                  style={{ left: hover[0], top: hover[1], width: 2 * brush, height: 2 * brush }}
                  className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.4)] pointer-events-none"
                />
              )}
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={() => setErasing(!erasing)} className={`${btn} ${erasing ? 'bg-gray-100 dark:bg-zinc-800' : ''}`}>
                <Eraser size={13} />
                {t('sam3.eraser')}
              </button>
              <button type="button" onClick={undo} disabled={!(erasing ? state.erase : state.points).length} className={btn}>
                <ArrowUUpLeft size={13} />
                {t('sam3.undo')}
              </button>
              <button type="button" onClick={clear} disabled={!state.points.length && !state.erase.length} className={btn}>
                <Trash size={13} />
                {t('sam3.clear')}
              </button>
            </div>
            {erasing && (
              <div className="group flex flex-col gap-1.5">
                <div className="flex justify-between">
                  <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.brush')}</span>
                  <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{2 * brush}px</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="40"
                  step="1"
                  value={brush}
                  onChange={(e) => setBrush(parseInt(e.target.value))}
                  className="w-full accent-[#0ea0ff] disabled:opacity-40"
                />
              </div>
            )}
          </div>

          <div className="group flex flex-col gap-1.5">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.preserve')}</span>
            <button type="button" onClick={() => set({ preserve: !state.preserve })} className="self-start">
              <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${state.preserve ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`}>
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${state.preserve ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </span>
            </button>
          </div>

          {/* Additive: every point is its own object, added (or, right-click, removed) on its own. */}
          <div className="group flex flex-col gap-1.5">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.additive')}</span>
            <button type="button" onClick={() => set({ additive: !state.additive })} className="self-start">
              <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${state.additive ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`}>
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${state.additive ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </span>
            </button>
          </div>

          {/* Text is the other SAM3 head: one shot behind Run, and only while no point is set. */}
          <div className="group flex flex-col gap-1">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.prompt')}</span>
            <div className="relative">
              <textarea
                value={state.text}
                disabled={state.points.length > 0}
                onChange={(e) => set({ text: e.target.value })}
                rows={2}
                className="input-textarea"
              />
              {state.text && !state.points.length && (
                <button type="button" onClick={() => set({ text: '' })} className="input-clear-btn">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

async function submit({ images, state }) {
  const fd = new FormData();
  fd.append('image', images[0]);
  fd.append('text', state.text);
  fd.append('preserve', state.preserve);
  const r = await apiPost('/sam3', fd);
  return { blob: await r.blob(), state };
}

export default {
  label: 'mode.sam3',
  // Loaded explicitly: the Run button reads Load model until then, and clicks wait for it.
  model: 'sam3',
  initialState,
  Inputs,
  submit,
  // Clicks are live, so Run is only for the text prompt — and only while no point is set.
  canSubmit: ({ images, state }) => images.length > 0 && !state.points.length && !!state.text.trim(),
  // The image itself is not persisted, so the points it was placed on are meaningless.
  restoreState: (state) => ({ ...state, points: [], labels: [], erase: [] }),
};
