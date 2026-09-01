import { useState, useEffect } from 'react';
import { X, Download, Image as ImageIcon } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';
import AdvancedSettings from '../AdvancedSettings';

const initialState = {
  prompt: '',
  numInferenceSteps: '',
  diffusionCoefficient: '',
  numImagesPerPrompt: '',
  seed: '',
  width: '',
  height: '',
};

function Inputs({ state, setState, images, setImages, onZoom }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <ImageDropZone images={images} onChange={setImages} multi onZoom={onZoom} />

      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.prompt')}</span>
        <div className="relative">
          <textarea
            value={state.prompt}
            onChange={(e) => set({ prompt: e.target.value })}
            rows={3}
            className="input-textarea"
          />
          {state.prompt && (
            <button type="button" onClick={() => set({ prompt: '' })} className="input-clear-btn">
              <X size={12} />
            </button>
          )}
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
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.diffusion_coefficient')}</span>
          <input
            type="number"
            step="0.1"
            placeholder="0"
            value={state.diffusionCoefficient}
            onChange={(e) => set({ diffusionCoefficient: e.target.value === '' ? '' : parseFloat(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.num_images')}</span>
          <input
            type="number"
            min="1"
            placeholder="1"
            value={state.numImagesPerPrompt}
            onChange={(e) => set({ numImagesPerPrompt: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.seed')}</span>
          <input
            type="number"
            placeholder="random"
            value={state.seed}
            onChange={(e) => set({ seed: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.width')}</span>
          <input
            type="number"
            placeholder="1024"
            value={state.width}
            onChange={(e) => set({ width: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.height')}</span>
          <input
            type="number"
            placeholder="1024"
            value={state.height}
            onChange={(e) => set({ height: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
      </div>
      </AdvancedSettings>
    </div>
  );
}

async function submit({ images, state }) {
  const fd = new FormData();
  for (const img of images) fd.append('images', img);
  fd.append('prompt', state.prompt);
  if (Number.isInteger(state.numInferenceSteps)) fd.append('num_inference_steps', state.numInferenceSteps);
  if (Number.isFinite(state.diffusionCoefficient)) fd.append('diffusion_coefficient', state.diffusionCoefficient);
  if (Number.isInteger(state.numImagesPerPrompt)) fd.append('num_images_per_prompt', state.numImagesPerPrompt);
  if (Number.isInteger(state.seed)) fd.append('seed', state.seed);
  if (Number.isInteger(state.width)) fd.append('width', state.width);
  if (Number.isInteger(state.height)) fd.append('height', state.height);
  const r = await apiPost('/flux2klein', fd);
  const { images: b64s } = await r.json();
  const blobs = await Promise.all(
    b64s.map((b) => fetch(`data:image/jpeg;base64,${b}`).then((res) => res.blob()))
  );
  return { meta: { images: blobs }, state };
}

const canSubmit = ({ state }) => !!state.prompt;

function Result({ meta, onZoom }) {
  const blobs = meta?.images || [];
  const [urls, setUrls] = useState([]);
  const [aspects, setAspects] = useState({});

  useEffect(() => {
    const created = blobs.map((b) => URL.createObjectURL(b));
    setUrls(created);
    setAspects({});
    return () => created.forEach(URL.revokeObjectURL);
  }, [meta]);

  if (!urls.length) {
    return (
      <div className="flex flex-col items-center gap-2 text-gray-300 dark:text-zinc-600">
        <ImageIcon size={40} />
      </div>
    );
  }

  const cols = Math.ceil(Math.sqrt(urls.length));
  const rows = Math.ceil(urls.length / cols);
  const rowHeight = `calc((100% - ${(rows - 1) * 8}px) / ${rows})`;
  const chunks = Array.from({ length: rows }, (_, r) => r * cols);
  const colWidth = `calc((100% - ${(cols - 1) * 8}px) / ${cols})`;

  return (
    <div className="w-full h-full flex flex-col justify-center gap-2 overflow-hidden">
      {chunks.map((offset) => (
        <div key={offset} className="w-full flex items-center justify-center gap-2 min-h-0" style={{ height: rowHeight }}>
          {urls.slice(offset, offset + cols).map((url, j) => {
            const i = offset + j;
            return (
              <div
                key={i}
                className="group/thumb relative"
                style={{ aspectRatio: aspects[i] ?? 1, maxHeight: '100%', maxWidth: colWidth }}
              >
                <img
                  src={url}
                  alt={`Generated ${i + 1}`}
                  onLoad={(e) => {
                    const a = e.currentTarget.naturalWidth / e.currentTarget.naturalHeight;
                    setAspects((prev) => (prev[i] === a ? prev : { ...prev, [i]: a }));
                  }}
                  onClick={() => onZoom(url)}
                  className="block w-full h-full object-contain rounded cursor-zoom-in"
                />
                <button
                  type="button"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `generated-${i + 1}.png`;
                    link.click();
                  }}
                  className="absolute top-1 right-1 flex items-center px-2 py-1 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 bg-white dark:bg-zinc-900 hover:bg-gray-50 dark:hover:bg-zinc-800 opacity-0 group-hover/thumb:opacity-100 transition-opacity"
                >
                  <Download size={13} />
                </button>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default {
  label: 'mode.flux2klein',
  initialState,
  Inputs,
  Result,
  submit,
  canSubmit,
};
