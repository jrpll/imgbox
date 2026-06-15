import { X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';
import { useLang } from '../../lib/i18n';
import ImageDropZone from '../ImageDropZone';
import AdvancedSettings from '../AdvancedSettings';

const initialState = {
  step: 1,
  trained: false,
  text1: '',
  text2: '',
  negPrompt: '',
  slider: 0.9,
  numTrainSteps: '',
  numInversionSteps: '',
};

function Inputs({ state, setState, images, setImages, onZoom }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  const locked = !state.trained;

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      {state.step === 1 ? (
        <>
          <ImageDropZone images={images} onChange={setImages} onZoom={onZoom} />

          <div className="group flex flex-col gap-1">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.source')}</span>
            <div className="relative">
              <textarea
                value={state.text1}
                onChange={(e) => set({ text1: e.target.value })}
                rows={3}
                className="input-textarea"
              />
              {state.text1 && (
                <button type="button" onClick={() => set({ text1: '' })} className="input-clear-btn">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <div className="group flex flex-col gap-1">
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.target')}</span>
            <div className="relative">
              <textarea
                value={state.text2}
                onChange={(e) => set({ text2: e.target.value })}
                rows={3}
                className="input-textarea"
              />
              {state.text2 && (
                <button type="button" onClick={() => set({ text2: '' })} className="input-clear-btn">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <AdvancedSettings>
          <div className="flex gap-3">
            <div className="group flex-1 flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.training_steps')}</span>
              <input
                type="number"
                placeholder="100"
                value={state.numTrainSteps}
                onChange={(e) => set({ numTrainSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
              />
            </div>
            <div className="group flex-1 flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.inversion_steps')}</span>
              <input
                type="number"
                placeholder="50"
                value={state.numInversionSteps}
                onChange={(e) => set({ numInversionSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
              />
            </div>
          </div>
          </AdvancedSettings>
        </>
      ) : (
        <>
          <div className={`${locked ? '' : 'group '}flex flex-col gap-1`}>
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.target')}</span>
            <div className="relative">
              <textarea
                value={state.text2}
                disabled={locked}
                onChange={(e) => set({ text2: e.target.value })}
                rows={3}
                className="input-textarea"
              />
              {state.text2 && !locked && (
                <button type="button" onClick={() => set({ text2: '' })} className="input-clear-btn">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <div className={`${locked ? '' : 'group '}flex flex-col gap-1`}>
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.negative_prompt')}</span>
            <div className="relative">
              <textarea
                value={state.negPrompt}
                disabled={locked}
                onChange={(e) => set({ negPrompt: e.target.value })}
                rows={3}
                className="input-textarea"
              />
              {state.negPrompt && !locked && (
                <button type="button" onClick={() => set({ negPrompt: '' })} className="input-clear-btn">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <div className={`${locked ? '' : 'group '}flex flex-col gap-1.5`}>
            <div className="flex justify-between">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.preservation')}</span>
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{Math.round((1 - state.slider) * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={state.slider}
              disabled={locked}
              onChange={(e) => set({ slider: parseFloat(e.target.value) })}
              className="w-full accent-[#0ea0ff] disabled:opacity-40 disabled:cursor-not-allowed"
            />
          </div>
        </>
      )}
    </div>
  );
}

async function submit({ images, state }) {
  if (state.step === 1) {
    const fd = new FormData();
    fd.append('image', images[0]);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    if (Number.isInteger(state.numTrainSteps)) fd.append('num_train_steps', state.numTrainSteps);
    if (Number.isInteger(state.numInversionSteps)) fd.append('num_inversion_steps', state.numInversionSteps);
    const r = await apiPost('/generate', fd);
    return { blob: await r.blob(), state: { ...state, trained: true, step: 2 } };
  } else {
    const fd = new FormData();
    fd.append('slider', state.slider);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    if (state.negPrompt) fd.append('neg_prompt', state.negPrompt);
    const r = await apiPost('/edit', fd);
    return { blob: await r.blob(), state };
  }
}

const canSubmit = ({ images, state }) => {
  if (state.step === 1) return !!(images.length && state.text1 && state.text2);
  return !!(state.trained && state.text2);
};

export default {
  label: 'mode.edit',
  initialState,
  Inputs,
  submit,
  canSubmit,
  totalSteps: 2,
  getStepLabel: (state, t) => state.step === 1 ? t('edit.step.training') : t('edit.step.edit'),
  restoreState: (state) => ({ ...state, step: 1, trained: false }),
};
