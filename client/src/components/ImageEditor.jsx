import { useState, useEffect, useRef } from 'react';
import { Upload, X, Download, ImageIcon } from 'lucide-react';
import { apiPost } from '../lib/api';
import boxIconRaw from '../assets/box.svg?raw'
import ThreeSpinner from './ThreeSpinner';

export default function ImageEditor() {
  const [image, setImage] = useState(null);
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditingSlider, setIsEditingSlider] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [sliderValue, setSliderValue] = useState(0.9);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };

  const handleDeleteImage = () => {
    setImage(null);
    setResult(null);
  };

  const handleReset = () => {
    setImage(null);
    setText1('');
    setText2('');
    setResult(null);
    setSliderValue(0.9);
  };

  useEffect(() => {
    let unlisten;
    (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        const { invoke } = await import('@tauri-apps/api/core');
        unlisten = await listen('tauri://drag-drop', async (event) => {
          const path = event.payload.paths?.[0];
          if (!path) return;
          const ext = path.split('.').pop().toLowerCase();
          const mimeMap = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', gif: 'image/gif', webp: 'image/webp', bmp: 'image/bmp' };
          if (!mimeMap[ext]) return;
          const bytes = await invoke('read_file', { path });
          const blob = new Blob([new Uint8Array(bytes)], { type: mimeMap[ext] });
          setImage(new File([blob], path.split('/').pop(), { type: mimeMap[ext] }));
          setIsDragging(false);
        });
      } catch { /* browser fallback */ }
    })();
    return () => { unlisten?.(); };
  }, []);

  const handleSubmit = async () => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('text1', text1);
    formData.append('text2', text2);
    try {
      const response = await apiPost('/generate', formData);
      const data = await response.blob();
      setResult(URL.createObjectURL(data));
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSliderChange = (e) => {
    const val = parseFloat(e.target.value);
    setSliderValue(val);
    if (result) {
      setIsEditingSlider(true);
      const formData = new FormData();
      formData.append('slider', val);
      formData.append('text1', text1);
      formData.append('text2', text2);
      apiPost('/edit', formData)
        .then(r => r.blob())
        .then(data => setResult(URL.createObjectURL(data)))
        .catch(err => console.error('Error:', err))
        .finally(() => setIsEditingSlider(false));
    }
  };

  const canRun = image && text1 && text2 && !isLoading;

  return (
    <div className="h-screen bg-white flex flex-col p-5 gap-3 overflow-hidden">

      {/* App header */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <h1 className="text-2xl font-bold">imgbox</h1>
        <span className="w-6 h-6 block" dangerouslySetInnerHTML={{ __html: boxIconRaw.replace(/width="\d+" height="\d+"/, 'width="24" height="24"') }} />
      </div>

      {/* Main row — two cards */}
      <div className="flex gap-4 flex-1 min-h-0">

        {/* Input card */}
        <div className="flex-1 flex flex-col rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-gray-100">
            <span className="font-semibold text-gray-800">Input</span>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">

            {/* Image upload */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              className={`relative h-40 flex items-center justify-center border-2 rounded-lg cursor-pointer overflow-hidden transition-colors flex-shrink-0 ${
                isDragging
                  ? 'border-blue-400 bg-blue-50'
                  : image
                    ? 'border-gray-200'
                    : 'border-dashed border-gray-300 hover:border-gray-400 bg-gray-50'
              }`}
            >
              {image ? (
                <>
                  <img src={URL.createObjectURL(image)} alt="Preview" className="max-w-[calc(100%-24px)] max-h-[calc(100%-24px)] object-contain rounded" />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); handleDeleteImage(); }}
                    className="absolute top-2 right-2 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
                  >
                    <X size={12} />
                  </button>
                </>
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-400">
                  <Upload size={24} />
                  <span className="text-sm">Click or drag an image</span>
                </div>
              )}
            </div>
            <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={handleImageChange} />

            {/* Source prompt */}
            <div className="relative">
              <textarea
                value={text1}
                placeholder="Source description..."
                onChange={(e) => setText1(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 text-gray-900 rounded-lg text-sm focus:outline-none focus:border-gray-400 resize-none placeholder-gray-400"
              />
              {text1 && (
                <button type="button" onClick={() => setText1('')} className="absolute top-2 right-2 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500">
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Target prompt */}
            <div className="relative">
              <textarea
                value={text2}
                placeholder="Target description..."
                onChange={(e) => setText2(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 text-gray-900 rounded-lg text-sm focus:outline-none focus:border-gray-400 resize-none placeholder-gray-400"
              />
              {text2 && (
                <button type="button" onClick={() => setText2('')} className="absolute top-2 right-2 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500">
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Slider */}
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

          {/* Buttons */}
          <div className="flex items-center gap-3 px-5 py-4 border-t border-gray-100">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-2 text-sm border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canRun}
              className="flex-1 py-2 text-sm font-medium rounded-md bg-[#0ea0ff] hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Processing...' : 'Run'}
            </button>
          </div>
        </div>

        {/* Result card */}
        <div className="flex-1 flex flex-col rounded-xl border border-gray-200 overflow-hidden">
          <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
            <span className="font-semibold text-gray-800">Result</span>
            {result && (
              <button
                type="button"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = result;
                  link.download = 'generated-image.png';
                  link.click();
                }}
                className="flex items-center gap-1.5 px-3 py-1 text-sm border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 transition-colors"
              >
                <Download size={13} />
                Download
              </button>
            )}
          </div>

          <div className="flex-1 flex items-center justify-center p-4 bg-gray-50">
            {isLoading ? (
              <div className="flex items-center gap-3 text-gray-400">
                <ThreeSpinner size={32} />
                <span className="text-sm">Processing...</span>
              </div>
            ) : result ? (
              <div className="relative w-full h-full">
                {isEditingSlider && (
                  <div className="absolute inset-0 bg-black/25 flex items-center justify-center z-10 rounded-lg">
                    <span className="text-white text-sm">Updating...</span>
                  </div>
                )}
                <img src={result} alt="Generated" className="w-full h-full object-contain rounded-lg" />
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-300">
                <ImageIcon size={40} />
                <span className="text-sm">Result will appear here</span>
              </div>
            )}
          </div>
        </div>
      </div>


</div>
  );
}
