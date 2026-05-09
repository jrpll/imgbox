import { useState, forwardRef, useImperativeHandle, useEffect } from 'react';
import { X } from 'lucide-react';
import { apiPost } from '../../lib/api';


const Edit = forwardRef(function Edit({ image, result, isLoading, onResult, onLoading, onEditingSlider, onReset, onCanRunChange }, ref) {
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [sliderValue, setSliderValue] = useState(0.9);

  const canRun = !!(image && text1 && text2 && !isLoading);

  useEffect(() => { onCanRunChange(canRun); }, [canRun]);

  const handleSubmit = async () => {
    onLoading(true);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('text1', text1);
    formData.append('text2', text2);
    try {
      const response = await apiPost('/generate', formData);
      onResult(URL.createObjectURL(await response.blob()));
    } catch (error) {
      console.error('Error:', error);
    } finally {
      onLoading(false);
    }
  };

  const handleReset = () => {
    setText1('');
    setText2('');
    setSliderValue(0.9);
    onReset();
  };

  useImperativeHandle(ref, () => ({ submit: handleSubmit, reset: handleReset }));

  const handleSliderChange = (e) => {
    const val = parseFloat(e.target.value);
    setSliderValue(val);
    if (result) {
      onEditingSlider(true);
      const formData = new FormData();
      formData.append('slider', val);
      formData.append('text1', text1);
      formData.append('text2', text2);
      apiPost('/edit', formData)
        .then(r => r.blob())
        .then(data => onResult(URL.createObjectURL(data)))
        .catch(err => console.error('Error:', err))
        .finally(() => onEditingSlider(false));
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <div className="relative">
        <textarea
          value={text1}
          placeholder="Source description..."
          onChange={(e) => setText1(e.target.value)}
          rows={3}
          className="input-textarea"
        />
        {text1 && (
          <button type="button" onClick={() => setText1('')} className="input-clear-btn">
            <X size={12} />
          </button>
        )}
      </div>

      <div className="relative">
        <textarea
          value={text2}
          placeholder="Target description..."
          onChange={(e) => setText2(e.target.value)}
          rows={3}
          className="input-textarea"
        />
        {text2 && (
          <button type="button" onClick={() => setText2('')} className="input-clear-btn">
            <X size={12} />
          </button>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">Structure preservation</span>
          <span className="text-sm text-gray-500">{Math.round(sliderValue * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={sliderValue}
          disabled={!result}
          onChange={handleSliderChange}
          className="w-full accent-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
        />
      </div>
    </div>
  );
});

export default Edit;
