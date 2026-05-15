import { X } from '@phosphor-icons/react';
import { apiPost } from '../../lib/api';

const initialState = {
  prompt: '',
  numInferenceSteps: '',
  diffusionCoefficient: '',
};

function Inputs({ state, setState }) {
  const set = (patch) => setState((s) => ({ ...s, ...patch }));

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <div className="group flex flex-col gap-1">
        <span className="text-xs text-gray-400 group-hover:text-gray-600">Prompt</span>
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

      <div className="flex gap-3">
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600">Inference steps</span>
          <input
            type="number"
            placeholder="100"
            value={state.numInferenceSteps}
            onChange={(e) => set({ numInferenceSteps: e.target.value === '' ? '' : parseInt(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 placeholder-gray-400"
          />
        </div>
        <div className="group flex-1 flex flex-col gap-1">
          <span className="text-xs text-gray-400 group-hover:text-gray-600">Diffusion coefficient</span>
          <input
            type="number"
            step="0.1"
            placeholder="3"
            value={state.diffusionCoefficient}
            onChange={(e) => set({ diffusionCoefficient: e.target.value === '' ? '' : parseFloat(e.target.value) })}
            className="w-full px-3 py-1.5 text-sm border border-gray-200 rounded focus:outline-none group-hover:border-gray-400 focus:border-gray-400 placeholder-gray-400"
          />
        </div>
      </div>
    </div>
  );
}

async function submit({ image, state }) {
  const fd = new FormData();
  fd.append('image', image);
  fd.append('prompt', state.prompt);
  if (state.numInferenceSteps !== '') fd.append('num_inference_steps', state.numInferenceSteps);
  if (state.diffusionCoefficient !== '') fd.append('diffusion_coefficient', state.diffusionCoefficient);
  const r = await apiPost('/flux2klein', fd);
  return { blob: await r.blob(), state };
}

const canSubmit = ({ image, state }) => !!(image && state.prompt);

export default {
  label: 'flux2 klein',
  initialState,
  Inputs,
  submit,
  canSubmit,
};
