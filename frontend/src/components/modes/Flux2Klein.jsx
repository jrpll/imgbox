import { X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';
import AdvancedSettings from '../AdvancedSettings';

const initialState = {
  prompt: '',
  numInferenceSteps: '',
  diffusionCoefficient: '',
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
  if (Number.isInteger(state.seed)) fd.append('seed', state.seed);
  if (Number.isInteger(state.width)) fd.append('width', state.width);
  if (Number.isInteger(state.height)) fd.append('height', state.height);
  const r = await apiPost('/flux2klein', fd);
  return { blob: await r.blob(), state };
}

const canSubmit = ({ state }) => !!state.prompt;

export default {
  label: 'mode.flux2klein',
  initialState,
  Inputs,
  submit,
  canSubmit,
};
