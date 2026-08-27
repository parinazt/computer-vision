import React, { useState, useRef } from 'react';
import { SAMPLE_IMAGES, MODEL_CATALOG } from '../data/modelsAndPresets';
import { SampleImage, ModelBackbone } from '../types';
import {
  Upload,
  Image as ImageIcon,
  Cpu,
  Layers,
  Zap,
  Activity,
  Eye,
  Sliders,
  Sparkles,
  Info,
  RefreshCw
} from 'lucide-react';

export const ClassifierPlayground: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<SampleImage>(SAMPLE_IMAGES[0]);
  const [customImage, setCustomImage] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<ModelBackbone>('resnet50');
  const [gradCamIntensity, setGradCamIntensity] = useState<number>(0.75);
  const [activeTab, setActiveTab] = useState<'predictions' | 'gradcam' | 'feature_maps'>('predictions');
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [top5Results, setTop5Results] = useState(SAMPLE_IMAGES[0].top5);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const modelInfo = MODEL_CATALOG[activeModel];

  const handleSelectSample = (sample: SampleImage) => {
    setCustomImage(null);
    setSelectedImage(sample);
    setTop5Results(sample.top5);
    triggerClassificationAnim(sample.top5);
  };

  const triggerClassificationAnim = (results: { label: string; prob: number }[]) => {
    setIsClassifying(true);
    setTimeout(() => {
      setTop5Results(results);
      setIsClassifying(false);
    }, 300);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      setCustomImage(url);

      const uploadedName = file.name.split('.')[0].replace(/[-_]/g, ' ');
      const synthesizedResults = [
        { label: `Custom Target (${uploadedName || 'Object'})`, prob: 0.942 },
        { label: 'Background Environment', prob: 0.038 },
        { label: 'Secondary Visual Element', prob: 0.012 },
        { label: 'Surface Texture Pattern', prob: 0.005 },
        { label: 'Ambient Edge Contour', prob: 0.003 },
      ];

      setSelectedImage({
        id: 'custom_uploaded',
        name: file.name,
        category: 'User Upload',
        url,
        predictedClass: uploadedName || 'Classified Object',
        confidence: 0.942,
        top5: synthesizedResults,
        gradCamDescription: 'Neural network activations highlight central high-contrast gradient clusters and spatial edge structures.',
        heatmapColor: 'fire',
      });

      triggerClassificationAnim(synthesizedResults);
    };
    reader.readAsDataURL(file);
  };

  const currentImageUrl = customImage || selectedImage.url;

  return (
    <div className="space-y-6">
      {/* Top Banner & Control Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 bg-zinc-950 border-2 border-white shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider bg-yellow-400 text-black border border-white">
              002 // PLAYGROUND
            </span>
            <span className="flex items-center gap-1.5 text-xs text-zinc-300 font-mono font-bold uppercase">
              <Activity className="w-3.5 h-3.5 text-yellow-400 stroke-[3]" /> LIVE INFERENCE SIMULATOR
            </span>
          </div>
          <h2 className="font-display text-xl sm:text-2xl font-black text-white uppercase tracking-tight mt-1">
            Vision Inference & Grad-CAM Heatmaps
          </h2>
          <p className="font-mono text-xs text-zinc-400 uppercase tracking-wide">
            Test classification backbones, inspect Top-5 softmax confidence, and analyze spatial activation maps.
          </p>
        </div>

        {/* Model Backbone Selector */}
        <div className="flex items-center gap-3 bg-black p-2 border-2 border-zinc-700">
          <Cpu className="w-4 h-4 text-yellow-400 ml-1 hidden sm:inline stroke-[2.5]" />
          <span className="text-xs font-mono font-black uppercase text-zinc-300">BACKBONE:</span>
          <select
            value={activeModel}
            onChange={(e) => {
              setActiveModel(e.target.value as ModelBackbone);
              triggerClassificationAnim(top5Results);
            }}
            className="bg-zinc-900 text-white text-xs font-mono font-bold px-3 py-1.5 border border-zinc-600 focus:outline-none focus:border-yellow-400 cursor-pointer"
          >
            {Object.values(MODEL_CATALOG).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Left Canvas / Image View, Right Analytics & Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Display, Grad-CAM Overlay, Samples (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-white text-black font-black flex items-center justify-center border border-white">
                  <ImageIcon className="w-3.5 h-3.5 stroke-[2.5]" />
                </div>
                <span className="font-display text-sm font-black uppercase tracking-wider text-white">
                  Input Stage
                </span>
                <span className="font-mono text-xs font-bold text-yellow-400">({modelInfo.inputDim})</span>
              </div>

              {/* View mode buttons */}
              <div className="flex bg-black p-1 border-2 border-zinc-700 text-xs font-mono">
                <button
                  onClick={() => setActiveTab('predictions')}
                  className={`px-3 py-1 font-black uppercase tracking-wider transition ${
                    activeTab === 'predictions' ? 'bg-yellow-400 text-black font-black' : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  RGB View
                </button>
                <button
                  onClick={() => setActiveTab('gradcam')}
                  className={`px-3 py-1 font-black uppercase tracking-wider transition flex items-center gap-1 ${
                    activeTab === 'gradcam' ? 'bg-yellow-400 text-black font-black' : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5 stroke-[2.5]" /> Grad-CAM
                </button>
                <button
                  onClick={() => setActiveTab('feature_maps')}
                  className={`px-3 py-1 font-black uppercase tracking-wider transition flex items-center gap-1 ${
                    activeTab === 'feature_maps' ? 'bg-yellow-400 text-black font-black' : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5 stroke-[2.5]" /> Filters
                </button>
              </div>
            </div>

            {/* Image Preview Container with GradCAM Effect */}
            <div className="relative aspect-video w-full border-2 border-zinc-700 bg-black overflow-hidden flex items-center justify-center group">
              <img
                src={currentImageUrl}
                alt={selectedImage.name}
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover object-center"
              />

              {/* Grad-CAM Heatmap Overlay */}
              {activeTab === 'gradcam' && (
                <div
                  className="absolute inset-0 pointer-events-none mix-blend-screen transition-opacity duration-200 flex items-center justify-center"
                  style={{
                    opacity: gradCamIntensity,
                    background:
                      selectedImage.heatmapColor === 'cyan'
                        ? 'radial-gradient(ellipse at 50% 50%, rgba(6,182,212,0.95) 0%, rgba(59,130,246,0.6) 45%, rgba(15,23,42,0) 80%)'
                        : selectedImage.heatmapColor === 'emerald'
                        ? 'radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.95) 0%, rgba(234,179,8,0.6) 45%, rgba(15,23,42,0) 80%)'
                        : 'radial-gradient(ellipse at 50% 50%, rgba(239,68,68,0.95) 0%, rgba(245,158,11,0.7) 40%, rgba(59,130,246,0.4) 65%, rgba(15,23,42,0) 85%)',
                  }}
                >
                  <div className="absolute top-3 left-3 bg-black text-yellow-400 border-2 border-yellow-400 px-3 py-1 text-xs font-mono font-black uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>Grad-CAM: Conv Layer Attention</span>
                  </div>
                </div>
              )}

              {/* Feature Maps Visualizer Overlay */}
              {activeTab === 'feature_maps' && (
                <div className="absolute inset-0 bg-black/95 p-4 flex flex-col justify-between">
                  <div className="flex items-center justify-between text-xs font-mono text-white border-b-2 border-zinc-800 pb-2">
                    <span className="font-bold uppercase tracking-wider">CONV FILTER CHANNELS (32 ACTIVATION SLICES)</span>
                    <span className="text-yellow-400 font-bold">56x56 PX</span>
                  </div>
                  <div className="grid grid-cols-8 gap-2 my-auto">
                    {Array.from({ length: 16 }).map((_, i) => (
                      <div
                        key={i}
                        className="aspect-square bg-zinc-900 border border-zinc-700 flex items-center justify-center text-[9px] font-mono text-zinc-400 overflow-hidden relative"
                      >
                        <div
                          className="w-full h-full opacity-80"
                          style={{
                            background: `radial-gradient(circle at ${(i * 37) % 100}% ${(i * 53) % 100}%, ${
                              i % 3 === 0 ? '#facc15' : i % 3 === 1 ? '#38bdf8' : '#ffffff'
                            } 0%, #000000 70%)`,
                          }}
                        />
                        <span className="absolute bottom-0.5 right-1 text-[8px] text-white font-mono font-bold">
                          f{i}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="text-[11px] font-mono text-zinc-400 uppercase">
                    Spatial activation weights capture edge orientation, corners, and texture gradients.
                  </p>
                </div>
              )}

              {/* Status Badge in bottom corner */}
              <div className="absolute bottom-3 right-3 bg-black text-white px-3 py-1 text-xs font-mono font-black uppercase border-2 border-white">
                {isClassifying ? (
                  <span className="flex items-center gap-1.5 text-yellow-400">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin stroke-[2.5]" /> INFERRING...
                  </span>
                ) : (
                  <span className="text-white">
                    TOP-1: <span className="text-yellow-400">{top5Results[0]?.label}</span>
                  </span>
                )}
              </div>
            </div>

            {/* Grad-CAM Intensity Slider (if active) */}
            {activeTab === 'gradcam' && (
              <div className="flex items-center gap-4 bg-black p-3 border-2 border-zinc-700 text-xs font-mono">
                <Sliders className="w-4 h-4 text-yellow-400 stroke-[2.5]" />
                <span className="text-white font-black uppercase">HEATMAP OPACITY:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={gradCamIntensity}
                  onChange={(e) => setGradCamIntensity(parseFloat(e.target.value))}
                  className="flex-1 accent-yellow-400 cursor-pointer"
                />
                <span className="font-mono font-black text-yellow-400 w-12 text-right">
                  {Math.round(gradCamIntensity * 100)}%
                </span>
              </div>
            )}

            {/* Gallery / Preset Sample Selector & Upload */}
            <div className="space-y-2 pt-3 border-t-2 border-zinc-800">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-black uppercase text-zinc-300">SELECT SAMPLE OR UPLOAD:</span>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1 text-yellow-400 hover:text-white font-black uppercase tracking-wider cursor-pointer border border-yellow-400 px-2 py-1 bg-black"
                >
                  <Upload className="w-3.5 h-3.5 stroke-[2.5]" /> UPLOAD IMAGE
                </button>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/*"
                  className="hidden"
                />
              </div>

              <div className="grid grid-cols-3 sm:grid-cols-6 gap-2.5">
                {SAMPLE_IMAGES.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    className={`relative border-2 aspect-square transition-all duration-150 cursor-pointer group ${
                      !customImage && selectedImage.id === sample.id
                        ? 'border-yellow-400 shadow-[3px_3px_0px_0px_rgba(250,204,21,1)]'
                        : 'border-zinc-800 hover:border-white opacity-70 hover:opacity-100'
                    }`}
                  >
                    <img
                      src={sample.url}
                      alt={sample.name}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/70 flex items-end p-1.5">
                      <span className="text-[10px] font-mono font-bold uppercase text-white truncate w-full text-left">
                        {sample.name}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Top-K Probabilities, Model Metrics & Explanations (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Top-5 Classification Predictions Card */}
          <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
            <div className="flex items-center justify-between border-b-2 border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-yellow-400 text-black font-black flex items-center justify-center border border-white">
                  <Zap className="w-3.5 h-3.5 stroke-[2.5]" />
                </div>
                <h3 className="font-display text-sm font-black uppercase tracking-tight text-white">
                  Softmax Probability (Top-5)
                </h3>
              </div>
              <span className="font-mono text-xs font-black text-zinc-400">SUM = 1.0</span>
            </div>

            {/* Probability Bars */}
            <div className="space-y-3">
              {top5Results.map((item, idx) => {
                const percentage = (item.prob * 100).toFixed(1);
                const isTop1 = idx === 0;

                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-6 h-5 flex items-center justify-center text-[10px] font-mono font-black border ${
                            isTop1
                              ? 'bg-yellow-400 text-black border-white'
                              : 'bg-black text-zinc-400 border-zinc-700'
                          }`}
                        >
                          #{idx + 1}
                        </span>
                        <span className={`uppercase font-bold ${isTop1 ? 'text-white font-black' : 'text-zinc-300'}`}>
                          {item.label}
                        </span>
                      </div>
                      <span className={`font-mono text-xs font-black ${isTop1 ? 'text-yellow-400' : 'text-zinc-400'}`}>
                        {percentage}%
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-3 bg-black border border-zinc-700 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${
                          isTop1
                            ? 'bg-yellow-400'
                            : idx === 1
                            ? 'bg-white'
                            : 'bg-zinc-600'
                        }`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Attention / Explanation text */}
            <div className="p-3 bg-black border-2 border-zinc-800 text-xs font-mono text-zinc-300 space-y-1.5">
              <div className="flex items-center gap-1.5 text-yellow-400 font-black uppercase text-[11px]">
                <Info className="w-3.5 h-3.5 stroke-[2.5]" />
                <span>SPATIAL ATTENTION SUMMARY</span>
              </div>
              <p className="text-zinc-400 leading-relaxed text-[11px] uppercase">
                {selectedImage.gradCamDescription}
              </p>
            </div>
          </div>

          {/* Architecture Benchmark Card */}
          <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
            <div className="flex items-center justify-between border-b-2 border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-white text-black font-black flex items-center justify-center border border-white">
                  <Cpu className="w-3.5 h-3.5 stroke-[2.5]" />
                </div>
                <h3 className="font-display text-sm font-black uppercase tracking-tight text-white">
                  Model Specs & Efficiency
                </h3>
              </div>
              <span className="px-2 py-0.5 text-[10px] font-mono font-black uppercase bg-white text-black border border-white">
                {modelInfo.type}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono">
              <div className="p-3 bg-black border border-zinc-700">
                <span className="text-[10px] uppercase font-black text-zinc-400">PARAMETERS</span>
                <div className="text-base font-black text-white mt-0.5">{modelInfo.parameters}</div>
                <span className="text-[10px] text-zinc-400 uppercase">Total weights</span>
              </div>

              <div className="p-3 bg-black border border-zinc-700">
                <span className="text-[10px] uppercase font-black text-zinc-400">COMPUTE LOAD</span>
                <div className="text-base font-black text-yellow-400 mt-0.5">{modelInfo.gflops}</div>
                <span className="text-[10px] text-zinc-400 uppercase">GFLOPs / Forward</span>
              </div>

              <div className="p-3 bg-black border border-zinc-700">
                <span className="text-[10px] uppercase font-black text-zinc-400">TOP-1 ACCURACY</span>
                <div className="text-base font-black text-white mt-0.5">{modelInfo.top1Accuracy}</div>
                <span className="text-[10px] text-zinc-400 uppercase">ImageNet Val</span>
              </div>

              <div className="p-3 bg-black border border-zinc-700">
                <span className="text-[10px] uppercase font-black text-zinc-400">CUDA LATENCY</span>
                <div className="text-base font-black text-yellow-400 mt-0.5">~12.4 ms</div>
                <span className="text-[10px] text-zinc-400 uppercase">FP16 TensorRT</span>
              </div>
            </div>

            <div className="text-[11px] font-mono uppercase text-zinc-400 bg-black p-3 border border-zinc-800">
              <span className="text-white font-black">RECOMMENDED APPLICATION: </span>
              {modelInfo.bestFor}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
