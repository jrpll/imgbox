import { apiPost } from '../../lib/api';

async function submit({ images, state }) {
  const fd = new FormData();
  fd.append('image', images[0]);
  const r = await apiPost('/remove-background', fd);
  return { blob: await r.blob(), state };
}

export default {
  label: 'mode.remove-background',
  maxImages: 1,
  initialState: {},
  Inputs: () => null,
  submit,
  canSubmit: ({ images }) => images.length > 0,
};
