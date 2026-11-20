import { X } from '@phosphor-icons/react';
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
  cfg: true,
};

function Inputs({ state, setState, images, setImages, onZoom }) {
  const { t } = useLang();
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  const locked = !state.trained;
  // Without CFG there is no uncond pass for the negative prompt to feed.
  const noNeg = locked || !state.cfg;

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
                placeholder={t('edit.source_example')}
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
                placeholder={t('edit.target_example')}
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
          <div className="flex flex-col gap-4">
            <div className="group flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.training_steps')}</span>
              <input
                type="number"
                placeholder="100"
                value={state.numTrainSteps}
                onChange={(e) => set({ numTrainSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
              />
            </div>
            <div className="group flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.inversion_steps')}</span>
              <input
                type="number"
                placeholder="50"
                value={state.numInversionSteps}
                onChange={(e) => set({ numInversionSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 dark:border-zinc-600 dark:bg-zinc-800 dark:text-gray-100 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 dark:group-hover:border-zinc-400 dark:focus:border-zinc-400 placeholder-gray-400 dark:placeholder-gray-500"
              />
            </div>
            {/* On: inversion and generation at CFG 7. Off: cond passes only, in both. */}
            <div className="group flex flex-col gap-1.5">
              <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.cfg')}</span>
              <button type="button" onClick={() => set({ cfg: !state.cfg })} className="self-start">
                <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${state.cfg ? 'bg-[#0ea0ff]' : 'bg-gray-300 dark:bg-zinc-600'}`}>
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${state.cfg ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </span>
              </button>
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

          <div className={`${noNeg ? '' : 'group '}flex flex-col gap-1`}>
            <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{t('edit.negative_prompt')}</span>
            <div className="relative">
              <textarea
                value={state.negPrompt}
                disabled={noNeg}
                onChange={(e) => set({ negPrompt: e.target.value })}
                rows={3}
                className="input-textarea"
              />
              {state.negPrompt && !noNeg && (
                <button type="button" onClick={() => set({ negPrompt: '' })} className="input-clear-btn">
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

// The preservation slider sits under the result it tunes, in the shell's result card.
function ResultControls({ state, setState, onRun }) {
  const { t } = useLang();
  if (state.step !== 2 || !state.trained) return null;
  return (
    <div className="group flex items-center gap-3 w-96 max-w-full">
      <span className="text-xs text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300 whitespace-nowrap">{t('edit.preservation')}</span>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={state.slider}
        onChange={(e) => setState((s) => ({ ...s, slider: parseFloat(e.target.value) }))}
        // Re-run on release, wherever the pointer ends up; by then the last value has rendered.
        onPointerDown={() => window.addEventListener('pointerup', onRun, { once: true })}
        className="flex-1 accent-[#0ea0ff]"
      />
      <span className="w-9 text-right text-xs tabular-nums text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">{Math.round((1 - state.slider) * 100)}%</span>
    </div>
  );
}

async function submit({ images, state, run }) {
  if (state.step === 1) {
    const fd = new FormData();
    fd.append('image', images[0]);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    if (Number.isInteger(state.numTrainSteps)) fd.append('num_train_steps', state.numTrainSteps);
    if (Number.isInteger(state.numInversionSteps)) fd.append('num_inversion_steps', state.numInversionSteps);
    fd.append('cfg', state.cfg);
    const r = await run('/generate', fd);
    return { blob: await r.blob(), state: { ...state, trained: true, step: 2 } };
  } else {
    const fd = new FormData();
    fd.append('slider', state.slider);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    if (state.negPrompt) fd.append('neg_prompt', state.negPrompt);
    fd.append('cfg', state.cfg);
    const r = await run('/edit', fd);
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
  tasks: ['edit'],
  model: 'edit',
  Inputs,
  ResultControls,
  submit,
  canSubmit,
  // Step 2 opens once there is a fine-tuned model to edit with.
  steps: [{ label: 'edit.step.training' }, { label: 'edit.step.edit', enabled: (state) => state.trained }],
  // The slider needs the fine-tuned transformer + inversion, which only live in server RAM.
  restoreState: (state, health) => (health.trained ? state : { ...state, step: 1, trained: false }),
};
