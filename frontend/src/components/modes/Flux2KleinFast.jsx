import { X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';
import AdvancedSettings from '../AdvancedSettings';
import flux2KleinMode from './Flux2Klein';

const initialState = {
  prompt: '',
  numImagesPerPrompt: '',
  seed: '',
  width: '',
  height: '',
};

const NUM_INPUT_CLASS = 'w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500';

function Inputs({ state, setState, images, setImages, onZoom }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));

  const numField = (key, labelKey, placeholder, extra = {}) => (
    <div className="group flex flex-col gap-1">
      <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t(labelKey)}</span>
      <input
        type="number"
        placeholder={placeholder}
        value={state[key]}
        onChange={(e) => set({ [key]: e.target.value === '' ? '' : parseInt(e.target.value) })}
        className={NUM_INPUT_CLASS}
        {...extra}
      />
    </div>
  );

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
        {numField('numImagesPerPrompt', 'flux.num_images', '1', { min: '1' })}
        {numField('seed', 'flux.seed', 'random')}
        {numField('width', 'flux.width', '1024')}
        {numField('height', 'flux.height', '1024')}
      </div>
      </AdvancedSettings>
    </div>
  );
}

async function submit({ images, state }) {
  const fd = new FormData();
  for (const img of images) fd.append('images', img);
  fd.append('prompt', state.prompt);
  if (Number.isInteger(state.numImagesPerPrompt)) fd.append('num_images_per_prompt', state.numImagesPerPrompt);
  if (Number.isInteger(state.seed)) fd.append('seed', state.seed);
  if (Number.isInteger(state.width)) fd.append('width', state.width);
  if (Number.isInteger(state.height)) fd.append('height', state.height);
  const r = await apiPost('/flux2klein-fast', fd);
  const { images: b64s } = await r.json();
  const blobs = await Promise.all(
    b64s.map((b) => fetch(`data:image/jpeg;base64,${b}`).then((res) => res.blob()))
  );
  return { meta: { images: blobs }, state };
}

const canSubmit = ({ state }) => !!state.prompt;

export default {
  label: 'mode.flux2klein-fast',
  initialState,
  Inputs,
  Result: flux2KleinMode.Result,
  submit,
  canSubmit,
};
