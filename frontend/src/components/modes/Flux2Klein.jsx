import { useState, useEffect } from 'react';
import { X, ArrowLeft, ArrowRight, Download, Image as ImageIcon } from '@phosphor-icons/react';
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
      <div className="flex gap-3">
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.inference_steps')}</span>
          <input
            type="number"
            placeholder="100"
            value={state.numInferenceSteps}
            onChange={(e) => set({ numInferenceSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.diffusion_coefficient')}</span>
          <input
            type="number"
            step="0.1"
            placeholder="3"
            value={state.diffusionCoefficient}
            onChange={(e) => set({ diffusionCoefficient: e.target.value === '' ? '' : parseFloat(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex-1 flex flex-col gap-1">
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
      </div>

      <div className="flex gap-3">
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.seed')}</span>
          <input
            type="number"
            placeholder="random"
            value={state.seed}
            onChange={(e) => set({ seed: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('flux.width')}</span>
          <input
            type="number"
            placeholder="1024"
            value={state.width}
            onChange={(e) => set({ width: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
          />
        </div>
        <div className="group flex-1 flex flex-col gap-1">
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
  const [index, setIndex] = useState(0);
  const [urls, setUrls] = useState([]);

  useEffect(() => {
    const created = blobs.map((b) => URL.createObjectURL(b));
    setUrls(created);
    setIndex((i) => Math.min(i, Math.max(0, created.length - 1)));
    return () => created.forEach(URL.revokeObjectURL);
  }, [meta]);

  if (!urls.length) {
    return (
      <div className="flex flex-col items-center gap-2 text-gray-300 dark:text-zinc-600">
        <ImageIcon size={40} />
      </div>
    );
  }

  const total = urls.length;
  const i = Math.min(index, total - 1);
  const url = urls[i];

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center gap-3">
      <button
        type="button"
        onClick={() => {
          const link = document.createElement('a');
          link.href = url;
          link.download = `generated-${i + 1}.png`;
          link.click();
        }}
        className="absolute top-0 right-0 z-10 flex items-center px-2 py-1 text-sm border border-gray-300 dark:border-zinc-600 rounded text-gray-600 dark:text-gray-300 bg-white dark:bg-zinc-900 hover:bg-gray-50 dark:hover:bg-zinc-800 transition-colors"
      >
        <Download size={13} />
      </button>
      <img
        src={url}
        alt="Generated"
        onClick={() => onZoom(url)}
        className="flex-1 min-h-0 w-full object-contain rounded cursor-zoom-in"
      />
      {total > 1 && (
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            type="button"
            onClick={() => setIndex((n) => Math.max(0, n - 1))}
            disabled={i === 0}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ArrowLeft size={15} />
          </button>
          <span className="text-xs text-gray-400 tabular-nums w-12 text-center">{i + 1} / {total}</span>
          <button
            type="button"
            onClick={() => setIndex((n) => Math.min(total - 1, n + 1))}
            disabled={i === total - 1}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ArrowRight size={15} />
          </button>
        </div>
      )}
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
