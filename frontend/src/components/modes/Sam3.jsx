import { useState, useEffect } from 'react';
import { X, ArrowUUpLeft, Trash } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';

const initialState = { points: [], labels: [], preserve: false, text: '' };

function Inputs({ state, setState, images, setImages, onZoom, onResult }) {
  const { t } = useLang();
  const [imageUrl, setImageUrl] = useState(null);
  const [cutout, setCutout] = useState(null);

  useEffect(() => {
    if (!images.length) { setImageUrl(null); return; }
    const url = URL.createObjectURL(images[0]);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [images]);

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
  }, [images, state.points, state.labels, state.preserve]);

  // Normalized coords so the points survive a resize and the persisted state stays
  // independent of the image resolution.
  const addPoint = (e, label) => {
    e.preventDefault();
    const box = e.currentTarget.getBoundingClientRect();
    setState((s) => ({
      ...s,
      points: [...s.points, [(e.clientX - box.left) / box.width, (e.clientY - box.top) / box.height]],
      labels: [...s.labels, label],
    }));
  };

  // Dropping the last point leaves nothing to show, so retract the live result too.
  const undo = () => setState((s) => {
    if (s.points.length === 1) onResult(null);
    return { ...s, points: s.points.slice(0, -1), labels: s.labels.slice(0, -1) };
  });
  const clear = () => { onResult(null); setState((s) => ({ ...s, points: [], labels: [] })); };

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
            <div
              onClick={(e) => addPoint(e, 1)}
              onContextMenu={(e) => addPoint(e, 0)}
              className="relative rounded overflow-hidden cursor-crosshair"
            >
              <img src={imageUrl} alt="Error loading image" className="block w-full" />
              {cutout && (
                <>
                  <span className="absolute inset-0 bg-[#0ea0ff] opacity-50 pointer-events-none" />
                  <img src={cutout} alt="" className="absolute inset-0 w-full h-full pointer-events-none" />
                </>
              )}
              {state.points.map(([x, y], i) => (
                <span
                  key={i}
                  style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
                  className={`absolute -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full border-2 border-white pointer-events-none ${
                    state.labels[i] ? 'bg-[#0ea0ff]' : 'bg-red-500'
                  }`}
                />
              ))}
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setImages([]); clear(); }}
                className="absolute top-1 right-1 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
              >
                <X size={12} />
              </button>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={undo} disabled={!state.points.length} className={btn}>
                <ArrowUUpLeft size={13} />
                {t('sam3.undo')}
              </button>
              <button type="button" onClick={clear} disabled={!state.points.length} className={btn}>
                <Trash size={13} />
                {t('sam3.clear')}
              </button>
            </div>
          </div>

          <div className="group flex flex-col gap-1.5">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('sam3.preserve')}</span>
            <button type="button" onClick={() => set({ preserve: !state.preserve })} className="self-start">
              <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${state.preserve ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`}>
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${state.preserve ? 'translate-x-4' : 'translate-x-0.5'}`} />
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
  initialState,
  Inputs,
  submit,
  // Clicks are live, so Run is only for the text prompt — and only while no point is set.
  canSubmit: ({ images, state }) => images.length > 0 && !state.points.length && !!state.text.trim(),
  // The image itself is not persisted, so the points it was placed on are meaningless.
  restoreState: (state) => ({ ...state, points: [], labels: [] }),
};
