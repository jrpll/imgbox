import { useEffect, useRef } from 'react';

export default function ThreeSpinner({ size = 300 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let renderer;
    let destroyed = false;

    (async () => {
      const THREE = await import('three/webgpu');
      const { Spinner, plotFunction, defaultConfig } = await import('../spinners/spinner.js');

      if (destroyed) return;

      renderer = new THREE.WebGPURenderer({ canvas: canvasRef.current, alpha: true, antialias: true });
      await renderer.init();

      if (destroyed) { renderer.dispose(); return; }

      renderer.setSize(size, size);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setClearColor(0x000000, 0);

      const scene = new THREE.Scene();
      const half = 0.35;
      const camera = new THREE.OrthographicCamera(-half, half, half, -half, 0.1, 10);
      camera.position.z = 1;

      const spinner = new Spinner(defaultConfig, plotFunction);
      scene.add(spinner);

      renderer.setAnimationLoop(() => renderer.render(scene, camera));
    })();

    return () => {
      destroyed = true;
      renderer?.setAnimationLoop(null);
      renderer?.dispose();
    };
  }, [size]);

  return <canvas ref={canvasRef} width={size} height={size} />;
}
