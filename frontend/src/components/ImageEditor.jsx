import { useState, useEffect, useRef } from 'react';
import { Upload, X, Download, Image, CaretDown } from '@phosphor-icons/react';
import boxIconRaw from '../assets/box.svg?raw'
import { apiPost } from '../lib/api';
import ThreeSpinner from './ThreeSpinner';
import editMode from './modes/Edit';
import removeBackgroundMode from './modes/RemoveBackground';

const MODES = {
  'edit': editMode,
  'remove-background': removeBackgroundMode,
};

export default function ImageEditor() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditingSlider, setIsEditingSlider] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [mode, setMode] = useState('edit');
  const [modeState, setModeState] = useState(MODES['edit'].initialState);
  const [menuOpen, setMenuOpen] = useState(false);
  const fileInputRef = useRef(null);
  const menuRef = useRef(null);

  const modeConfig = MODES[mode];
  const canRun = !isLoading && modeConfig.canSubmit({ image, state: modeState });

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleModeChange = (value) => {
    setMode(value);
    setModeState(MODES[value].initialState);
    setMenuOpen(false);
    setImage(null);
    setResult(null);

    const fd = new FormData();
    fd.append('name', value);
    apiPost('/mode', fd).catch((err) => console.error('mode swap:', err));
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      const blobUrl = await modeConfig.submit({ image, state: modeState });
      setResult(blobUrl);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setModeState(modeConfig.initialState);
    setImage(null);
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setImage(file);
    setIsDragging(false);
  };

  const Inputs = modeConfig.Inputs;

  return (
    <div className="h-screen bg-white flex flex-col p-5 gap-3 overflow-hidden">

      {/* Header */}
      <div className="flex items-baseline gap-2 flex-shrink-0">
        <h1 className="text-2xl font-bold">imgbox</h1>
        <span className="w-6 h-6 block self-center" dangerouslySetInnerHTML={{ __html: boxIconRaw.replace(/width="\d+" height="\d+"/, 'width="24" height="24"') }} />
        <div className="relative ml-2" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(o => !o)}
            className="flex items-center gap-2 px-4 py-1.5 text-base font-normal leading-none border border-gray-300 rounded hover:bg-gray-50 transition-colors min-w-[180px] justify-between"
            style={{ textBox: 'trim-both cap alphabetic' }}
          >
            {modeConfig.label}
            <CaretDown size={14} className={`transition-transform ${menuOpen ? 'rotate-180' : ''}`} />
          </button>
          {menuOpen && (
            <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded shadow-lg z-50 min-w-[280px] py-1">
              {Object.entries(MODES).map(([value, cfg]) => (
                <button
                  key={value}
                  onClick={() => handleModeChange(value)}
                  className={`w-full text-left px-4 py-2 text-base font-normal transition-colors ${mode === value ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
                >
                  {cfg.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main row — two cards */}
      <div className="flex gap-4 flex-1 min-h-0">

        {/* Input card */}
        <div className="flex-1 flex flex-col rounded border border-gray-200 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-gray-100">
            <span className="font-semibold">Input</span>
          </div>

          {/* Image upload — shared across all modes */}
          <div className="px-5 pt-4 flex-shrink-0">
            <div
              onClick={() => fileInputRef.current?.click()}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`relative h-40 flex items-center justify-center rounded cursor-pointer overflow-hidden transition-colors ${
                isDragging ? 'bg-blue-50' : image ? '' : 'bg-gray-50'
              }`}
              style={{
                backgroundImage: isDragging
                  ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%2360a5fa' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                  : image
                    ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23e5e7eb' stroke-width='2' stroke-dasharray='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                    : isHovered
                      ? `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%239ca3af' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
                      : `url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='4' ry='4' stroke='%23d1d5db' stroke-width='2' stroke-dasharray='6 5' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e")`
              }}
            >
              {image ? (
                <>
                  <img src={URL.createObjectURL(image)} alt="Preview" className="max-w-[calc(100%-24px)] max-h-[calc(100%-24px)] object-contain rounded" />
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setImage(null); setResult(null); }}
                    className="absolute top-2 right-2 p-0.5 bg-gray-400 text-white rounded-full hover:bg-gray-500"
                  >
                    <X size={12} />
                  </button>
                </>
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-400">
                  <Upload size={24} />
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(e) => { const f = e.target.files[0]; if (f) setImage(f); }}
            />
          </div>

          {/* Mode-specific inputs */}
          <div className="flex-1 flex flex-col min-h-0">
            <Inputs
              state={modeState}
              setState={setModeState}
              result={result}
              onResult={setResult}
              onEditingSlider={setIsEditingSlider}
            />
          </div>

          {/* Shared action buttons */}
          <div className="flex items-center gap-3 px-5 py-4 border-t border-gray-100">
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-2 text-sm border border-gray-300 rounded text-gray-600 hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canRun}
              className="flex-1 py-2 text-sm font-medium rounded bg-[#0ea0ff] hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Processing...' : 'Run'}
            </button>
          </div>
        </div>

        {/* Result card */}
        <div className="flex-1 flex flex-col rounded border border-gray-200 overflow-hidden">
          <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
            <span className="font-semibold">Result</span>
            {result && (
              <button
                type="button"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = result;
                  link.download = 'generated-image.png';
                  link.click();
                }}
                className="flex items-center gap-1.5 px-3 py-1 text-sm border border-gray-300 rounded text-gray-600 hover:bg-gray-50 transition-colors"
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
                  <div className="absolute inset-0 bg-black/25 flex items-center justify-center z-10 rounded">
                    <span className="text-white text-sm">Updating...</span>
                  </div>
                )}
                <img src={result} alt="Generated" className="w-full h-full object-contain rounded" />
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-300">
                <Image size={40} />
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}
