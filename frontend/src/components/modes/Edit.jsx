import { X } from 'lucide-react';
import { apiPost } from '../../lib/api';

const initialState = { text1: '', text2: '', slider: 0.9 };

function Inputs({ state, setState, result, onResult, onEditingSlider }) {
  const set = (patch) => setState((s) => ({ ...s, ...patch }));

  const handleSliderChange = async (e) => {
    const val = parseFloat(e.target.value);
    set({ slider: val });
    if (!result) return;

    onEditingSlider(true);
    try {
      const fd = new FormData();
      fd.append('slider', val);
      fd.append('text1', state.text1);
      fd.append('text2', state.text2);
      const r = await apiPost('/edit', fd);
      onResult(URL.createObjectURL(await r.blob()));
    } catch (err) {
      console.error('Error:', err);
    } finally {
      onEditingSlider(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <div className="relative">
        <textarea
          value={state.text1}
          placeholder="Source description..."
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

      <div className="relative">
        <textarea
          value={state.text2}
          placeholder="Target description..."
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

      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">Structure preservation</span>
          <span className="text-sm text-gray-500">{Math.round(state.slider * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={state.slider}
          disabled={!result}
          onChange={handleSliderChange}
          className="w-full accent-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
        />
      </div>
    </div>
  );
}

async function submit({ image, state }) {
  const fd = new FormData();
  fd.append('image', image);
  fd.append('text1', state.text1);
  fd.append('text2', state.text2);
  const r = await apiPost('/generate', fd);
  return URL.createObjectURL(await r.blob());
}

const canSubmit = ({ image, state }) => !!(image && state.text1 && state.text2);

export default {
  label: 'edit',
  initialState,
  Inputs,
  submit,
  canSubmit,
};
