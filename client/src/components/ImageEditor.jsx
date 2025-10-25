import React, { useState, useEffect, useRef } from 'react';
import { Upload, X, Download } from 'lucide-react';
import boxIcon from '../assets/box.svg'  // Add this import at the top

export default function ImageEditor() {
  const [image, setImage] = useState(null);
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditingSlider, setIsEditingSlider] = useState(false);
  const [scale, setScale] = useState(1);
  const fileInputRef = useRef(null);
  const BOX_SIZE = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--box-size'));
  const BOX_GAP = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--box-gap'));

  useEffect(() => {
    const calculateScale = () => {
      const viewportHeight = window.innerHeight;
      const availableHeight = viewportHeight - 150;
      const scale = Math.min(1, availableHeight / BOX_SIZE);
      setScale(scale);
    };

    calculateScale();
    window.addEventListener('resize', calculateScale);
    return () => window.removeEventListener('resize', calculateScale);
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  const handleDeleteImage = () => {
    setImage(null);
    setResult(null);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setIsLoading(true);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('text1', text1);
    formData.append('text2', text2);

    try {
      const response = await fetch('http://35.208.53.156:8080/generate', {
        method: 'POST',
        body: formData,
      });
      const data = await response.blob();
      setResult(URL.createObjectURL(data));
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const imageContainerStyle = {
    width: 'var(--box-size)',
    height: 'var(--box-size)',
    transform: `scale(${scale})`,
    transformOrigin: 'top left'
  };

  const wrapperStyle = {
    width: `${BOX_SIZE * scale}px`,
    height: `${BOX_SIZE * scale}px`,
  };

  return (
    <div className="min-h-screen bg-white overflow-x-hidden">
      <div className="flex justify-center mb-2 pt-2">
        <div style={{ width: `${BOX_SIZE * scale * 2 + BOX_GAP}px` }}>
          <div className="flex items-center gap-3">
            <h1 className="text-4xl font-bold">imgbox</h1>
            <img src={boxIcon} alt="imgbox logo" className="w-10 h-10" />
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        {/* Images Section */}
        <form>
          <div className="flex gap-4 justify-center">
            {/* Input Image Section */}
            <div style={wrapperStyle}>
              <div className="relative">
                <div 
                  onClick={handleUploadClick}
                  style={imageContainerStyle}
                  className="flex items-center justify-center border-2 border-gray-500 rounded-lg cursor-pointer overflow-hidden"
                >
                  {image ? (
                    <div className="relative w-full h-full">
                      <img 
                        src={URL.createObjectURL(image)} 
                        alt="Preview" 
                        className="w-full h-full object-contain"
                      />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteImage();
                        }}
                        className="absolute top-2 right-2 p-2 bg-gray-500 text-white rounded-full hover:bg-gray-600"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ) : (
                    <div className="text-center">
                      <Upload className="mx-auto h-20 w-20 text-gray-400" />
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageChange}
                  required
                />
              </div>
            </div>

            {/* Result Image Section */}
            <div style={wrapperStyle}>
              <div 
                style={imageContainerStyle}
                className="flex items-center justify-center border-2 border-gray-500 rounded-lg overflow-hidden"
              >
                {isLoading ? (
                  <div className="text-center">
                    <span className="block text-sm text-gray-600">Processing...</span>
                  </div>
                ) : result ? (
                  <div className="relative w-full h-full">
                    {isEditingSlider && ( // ← ADD THESE 3 LINES
                      <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
                        <span className="text-white text-lg">editing...</span>
                      </div>
                    )}
                    <img 
                      src={result} 
                      alt="Generated" 
                      className="w-full h-full object-contain"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = result;
                        link.download = 'generated-image.png';
                        link.click();
                      }}

                      className="absolute top-2 right-2 p-2 bg-gray-500 text-white rounded-full hover:bg-gray-900"
                    >
                      <Download size={16} />
                    </button>
                  </div>
                ) : (
                  <div className="text-center">
                    <span className="block text-gray-400 text-3xl">resultado</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </form>

        {/* Text inputs in a separate container */}
        {/* Text inputs in a separate container */}
<div className="flex justify-center items-center gap-4 mt-2" style={{ position: 'relative', zIndex: 50 }}>
  <div className="flex items-center gap-4" style={{ width: `${BOX_SIZE * scale * 2 + BOX_GAP}px` }}>
    <div className="relative w-1/3">
      <input
        type="text"
        value={text1}
        placeholder="descripcion de la imagen original"
        onChange={(e) => setText1(e.target.value)}
        className="w-full h-10 px-3 py-2 border border-gray-500 text-gray-900 rounded-lg text-base focus:outline-none focus:border-gray-500"
        required
      />
      {text1 && (
        <button
          type="button"
          onClick={() => setText1('')}
          className="absolute top-1/2 right-2 p-1 bg-gray-500 text-white rounded-full hover:bg-gray-600 transform -translate-y-1/2"
        >
          <X size={20} />
        </button>
      )}
    </div>
    <div className="relative w-1/3">
      <input
        type="text"
        value={text2}
        placeholder="descripcion de la imagen editada"
        onChange={(e) => setText2(e.target.value)}
        className="w-full h-10 px-3 py-2 border border-gray-500 text-gray-900 rounded-lg text-base focus:outline-none focus:border-gray-500"
        required
      />
      {text2 && (
        <button
          type="button"
          onClick={() => setText2('')}
          className="absolute top-1/2 right-2 p-1 bg-gray-500 text-white rounded-full hover:bg-gray-600 transform -translate-y-1/2"
        >
          <X size={20} />
        </button>
      )}
    </div>
    <button
      onClick={handleSubmit}
      disabled={!image || !text1 || !text2 || isLoading}
      className="w-1/3 h-10 flex justify-center items-center px-4 border border-transparent rounded-md shadow-sm text-base font-medium text-white bg-[#0ea0ff] hover:bg-blue-900 focus:outline-none focus:ring-2 focus:ring-offset-2"
    >
      {isLoading ? 'Processing...' : 'editar'}
    </button>
  </div>
</div>
        {/* Slider section */}
        <div className="flex justify-center items-center gap-4 mt-3" style={{ position: 'relative', zIndex: 50 }}>
          <div className="w-2/3 flex items-center gap-4">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              defaultValue="0.9"
              disabled={!result}
              className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer"
              onChange={(e) => {
                if (result) {
                  setIsEditingSlider(true);
		  const formData = new FormData();
                  formData.append('slider', e.target.value);
		  formData.append('text1', text1);  // ← ADD THIS
    		  formData.append('text2', text2);                  
                  fetch('http://35.208.53.156:8080/edit', {
                    method: 'POST',
                    body: formData,
                  })
                  .then(response => response.blob())
                  .then(data => {
                    setResult(URL.createObjectURL(data));
                  })
                  .catch(error => console.error('Error:', error))
                  .finally(() => setIsEditingSlider(false));
		}
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

