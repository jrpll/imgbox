import { X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';

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

function Inputs({ state, setState }) {
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  const locked = !state.trained;

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      {state.step === 1 ? (
        <>
          <div className="group flex flex-col gap-1">
            <span className="text-xs text-gray-400 group-hover:text-gray-600">Source description</span>
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
            <span className="text-xs text-gray-400 group-hover:text-gray-600">Target description</span>
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

          <div className="flex gap-3">
            <div className="group flex-1 flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600">Training steps</span>
              <input
                type="number"
                placeholder="100"
                value={state.numTrainSteps}
                onChange={(e) => set({ numTrainSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 placeholder-gray-400"
              />
            </div>
            <div className="group flex-1 flex flex-col gap-1">
              <span className="text-xs text-gray-400 group-hover:text-gray-600">Inversion steps</span>
              <input
                type="number"
                placeholder="50"
                value={state.numInversionSteps}
                onChange={(e) => set({ numInversionSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
                className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 placeholder-gray-400"
              />
            </div>
          </div>
        </>
      ) : (
        <>
          <div className={`${locked ? '' : 'group '}flex flex-col gap-1`}>
            <span className="text-xs text-gray-400 group-hover:text-gray-600">Target description</span>
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
            <span className="text-xs text-gray-400 group-hover:text-gray-600">Negative prompt</span>
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
              <span className="text-xs text-gray-400 group-hover:text-gray-600">Image preservation</span>
              <span className="text-xs text-gray-400 group-hover:text-gray-600">{Math.round((1 - state.slider) * 100)}%</span>
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

async function submit({ image, state, setState }) {
  if (state.step === 1) {
    const fd = new FormData();
    fd.append('image', image);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    fd.append('num_train_steps', state.numTrainSteps);
    fd.append('num_inversion_steps', state.numInversionSteps);
    const r = await apiPost('/generate', fd);
    const url = URL.createObjectURL(await r.blob());
    setState((s) => ({ ...s, trained: true, step: 2 }));
    return url;
  } else {
    const fd = new FormData();
    fd.append('slider', state.slider);
    fd.append('text1', state.text1);
    fd.append('text2', state.text2);
    if (state.negPrompt) fd.append('neg_prompt', state.negPrompt);
    const r = await apiPost('/edit', fd);
    return URL.createObjectURL(await r.blob());
  }
}

const canSubmit = ({ image, state }) => {
  if (state.step === 1) return !!(image && state.text1 && state.text2 && state.numTrainSteps && state.numInversionSteps);
  return !!(state.trained && state.text2);
};

export default {
  label: 'edit',
  initialState,
  Inputs,
  submit,
  canSubmit,
  totalSteps: 2,
  getStepLabel: (state) => state.step === 1 ? 'Training' : 'Edit',
};
