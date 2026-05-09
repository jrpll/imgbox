import { attribute, float, PI2, time, vec3, select, hash } from "three/tsl";
import * as THREE from "three/webgpu";

// Tailwind gray-400 (#9ca3af) in linear RGB
const gray400 = vec3(0.612, 0.639, 0.686);

export class Spinner extends THREE.Points {
  constructor(config, plotFn) {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsNodeMaterial({
      blending: THREE.NormalBlending,
      transparent: true,
    });
    super(geometry, material);
    this.config = config;
    this.plotFn = plotFn;
    this.rebuild(config);
  }

  rebuild(config) {
    const pointsPerParticle = config.strokeWidth;
    const totalParticles = pointsPerParticle * config.particleCount;

    const indices = new Float32Array(totalParticles);
    for (let i = 0; i < totalParticles; i++) indices[i] = i;

    this.geometry.setAttribute("indexAttr", new THREE.BufferAttribute(indices, 1));
    this.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(totalParticles * 3), 3));

    const pointIndex = attribute("indexAttr");
    const particleIndex = pointIndex.toFloat().div(pointsPerParticle).floor();
    const progress = particleIndex.div(config.particleCount);
    const origin = this.plotFn(progress, float(0.7).add(time.mul(3).sin().mul(0.02)), config);

    const animatedProgress = time.div(4).mod(1);
    const trailLength = float(0.3);
    const animationGradient = progress.sub(animatedProgress).add(1).mod(1);
    const insideTrail = animationGradient.lessThanEqual(trailLength);
    const gradient = select(insideTrail, animationGradient.add(0.1), float(config.strokeWidth * 0.7));

    const rand = hash(particleIndex);
    const length = float(config.strokeWidth).mul(gradient).mul(pointIndex.toFloat().mod(14).div(14)).mul(0.3);
    const ang = PI2.mul(rand);

    const mat = this.material;
    mat.positionNode = origin.add(vec3(ang.cos(), ang.sin(), 0).mul(length)).mul(float(1).add(time.sin().mul(0.01)));
    mat.colorNode = gray400;
    mat.needsUpdate = true;
    mat.opacityNode = gradient.add(0.2).mul(animationGradient.div(2));
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}

export const plotFunction = (progress, detailScale, config) => {
  const t = PI2.mul(progress);
  const R = float(config.spiralR);
  const r = float(config.spiralr);
  const d = float(config.spirald).add(detailScale.mul(0.25));
  const diff = R.sub(r);
  const ratio = diff.div(r);
  const baseX = diff.mul(t.cos()).add(d.mul(t.mul(ratio).cos()));
  const baseY = diff.mul(t.sin()).sub(d.mul(t.mul(ratio).sin()));
  const scale = float(config.spiralScale).add(detailScale.mul(config.spiralBreath));
  return vec3(baseX.mul(scale), baseY.mul(scale), 0);
};

export const defaultConfig = {
  strokeWidth: 0.3,
  particleCount: 100000,
  spiralR: 0.3,
  spiralr: 0.1,
  spirald: 0.1,
  spiralScale: 0.5,
  spiralBreath: 0.15,
};
