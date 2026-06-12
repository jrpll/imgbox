import { apiPost } from '../../lib/api';
import ImageDropZone from '../ImageDropZone';

function Inputs({ images, setImages, onZoom }) {
  return (
    <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
      <ImageDropZone images={images} onChange={setImages} onZoom={onZoom} />
    </div>
  );
}

async function submit({ images, state }) {
  const fd = new FormData();
  fd.append('image', images[0]);
  const r = await apiPost('/remove-background', fd);
  return { blob: await r.blob(), state };
}

export default {
  label: 'mode.remove-background',
  initialState: {},
  Inputs,
  submit,
  canSubmit: ({ images }) => images.length > 0,
};
