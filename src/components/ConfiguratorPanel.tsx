import React from 'react';
import { CodeGeneratorConfig, Framework, ModelBackbone, OptimizerType, DeviceType, TaskType } from '../types';
import { MODEL_CATALOG } from '../data/modelsAndPresets';
import { Sliders, Cpu, Database, Settings2, Sparkles, ShieldCheck } from 'lucide-react';

interface ConfiguratorPanelProps {
  config: CodeGeneratorConfig;
  onChange: (newConfig: CodeGeneratorConfig) => void;
  activeView: 'code' | 'requirements';
  setActiveView: (view: 'code' | 'requirements') => void;
}

export const ConfiguratorPanel: React.FC<ConfiguratorPanelProps> = ({
  config,
  onChange,
  activeView,
  setActiveView,
}) => {
  const updateConfig = (patch: Partial<CodeGeneratorConfig>) => {
    onChange({ ...config, ...patch });
  };

  return (
    <div className="bg-zinc-950 border-2 border-white p-5 sm:p-6 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b-2 border-zinc-800 pb-4 gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-white text-black font-black flex items-center justify-center border border-white">
            <Sliders className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div>
            <h3 className="font-display text-sm sm:text-base font-black uppercase tracking-tight text-white">
              001 // PIPELINE SPECIFICATIONS & PARAMETERS
            </h3>
            <p className="font-mono text-[10px] text-zinc-400 uppercase tracking-widest">
              Configurable PyTorch, TensorFlow & OpenCV architecture generator
            </p>
          </div>
        </div>
        
        {/* Output view toggle */}
        <div className="flex bg-black p-1 border-2 border-zinc-700 text-xs font-mono">
          <button
            onClick={() => setActiveView('code')}
            className={`px-3 py-1.5 font-black uppercase tracking-wider transition cursor-pointer border ${
              activeView === 'code'
                ? 'bg-yellow-400 text-black border-yellow-400'
                : 'bg-transparent text-zinc-400 border-transparent hover:text-white'
            }`}
          >
            Python Script (.py)
          </button>
          <button
            onClick={() => setActiveView('requirements')}
            className={`px-3 py-1.5 font-black uppercase tracking-wider transition cursor-pointer border ${
              activeView === 'requirements'
                ? 'bg-yellow-400 text-black border-yellow-400'
                : 'bg-transparent text-zinc-400 border-transparent hover:text-white'
            }`}
          >
            requirements.txt
          </button>
        </div>
      </div>

      {/* Grid of options */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Framework Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-black uppercase tracking-wider text-zinc-300 flex items-center justify-between">
            <span>Framework / Engine</span>
          </label>
          <select
            value={config.framework}
            onChange={(e) => {
              const fw = e.target.value as Framework;
              let defaultTask: TaskType = 'transfer_learning';
              if (fw === 'opencv_webcam') defaultTask = 'inference_only';
              updateConfig({ framework: fw, taskType: defaultTask });
            }}
            className="w-full bg-black text-white text-xs font-mono font-bold px-3 py-2.5 border-2 border-zinc-700 focus:border-yellow-400 focus:outline-none cursor-pointer"
          >
            <option value="pytorch">PyTorch (Torchvision)</option>
            <option value="tensorflow">TensorFlow / Keras</option>
            <option value="huggingface">Hugging Face (Transformers ViT)</option>
            <option value="sklearn">Scikit-Learn (OpenCV HOG + SVM)</option>
            <option value="opencv_webcam">OpenCV Live Webcam Classifier</option>
          </select>
        </div>

        {/* Task / Mode */}
        <div className="space-y-1.5">
          <label className="text-xs font-black uppercase tracking-wider text-zinc-300">
            Task Pipeline
          </label>
          <select
            value={config.taskType}
            onChange={(e) => updateConfig({ taskType: e.target.value as TaskType })}
            disabled={config.framework === 'opencv_webcam' || config.framework === 'sklearn'}
            className="w-full bg-black text-white text-xs font-mono font-bold px-3 py-2.5 border-2 border-zinc-700 focus:border-yellow-400 focus:outline-none cursor-pointer disabled:opacity-40"
          >
            <option value="transfer_learning">Pretrained Transfer Learning & Fine-Tuning</option>
            <option value="custom_cnn">Custom CNN From Scratch</option>
            <option value="inference_only">Inference / Predict Single Image Script</option>
          </select>
        </div>

        {/* Model Backbone */}
        <div className="space-y-1.5">
          <label className="text-xs font-black uppercase tracking-wider text-zinc-300">
            Model Backbone Architecture
          </label>
          <select
            value={config.modelBackbone}
            onChange={(e) => updateConfig({ modelBackbone: e.target.value as ModelBackbone })}
            disabled={config.framework === 'sklearn' || config.taskType === 'custom_cnn'}
            className="w-full bg-black text-white text-xs font-mono font-bold px-3 py-2.5 border-2 border-zinc-700 focus:border-yellow-400 focus:outline-none cursor-pointer disabled:opacity-40"
          >
            {Object.values(MODEL_CATALOG).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {/* Number of Classes */}
        <div className="space-y-1.5">
          <label className="text-xs font-black uppercase tracking-wider text-zinc-300 flex items-center justify-between">
            <span>Target Classes</span>
            <span className="font-mono font-black text-yellow-400 bg-zinc-900 px-1.5 py-0.5 border border-zinc-700">
              {config.numClasses} CLS
            </span>
          </label>
          <input
            type="number"
            min="2"
            max="1000"
            value={config.numClasses}
            onChange={(e) => updateConfig({ numClasses: Math.max(2, parseInt(e.target.value) || 2) })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-3 py-2 border-2 border-zinc-700 focus:border-yellow-400 focus:outline-none"
          />
        </div>
      </div>

      {/* Second row of hyperparameters */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-4 border-t-2 border-zinc-800">
        {/* Batch Size */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">Batch Size</span>
          <select
            value={config.batchSize}
            onChange={(e) => updateConfig({ batchSize: parseInt(e.target.value) })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-2.5 py-1.5 border border-zinc-700 focus:border-white focus:outline-none"
          >
            <option value="16">16</option>
            <option value="32">32 (Default)</option>
            <option value="64">64</option>
            <option value="128">128</option>
          </select>
        </div>

        {/* Epochs */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">Epochs</span>
          <select
            value={config.epochs}
            onChange={(e) => updateConfig({ epochs: parseInt(e.target.value) })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-2.5 py-1.5 border border-zinc-700 focus:border-white focus:outline-none"
          >
            <option value="5">5</option>
            <option value="10">10</option>
            <option value="20">20 (Recommended)</option>
            <option value="50">50</option>
          </select>
        </div>

        {/* Optimizer */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">Optimizer</span>
          <select
            value={config.optimizer}
            onChange={(e) => updateConfig({ optimizer: e.target.value as OptimizerType })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-2.5 py-1.5 border border-zinc-700 focus:border-white focus:outline-none"
          >
            <option value="AdamW">AdamW</option>
            <option value="Adam">Adam</option>
            <option value="SGD">SGD (Momentum)</option>
          </select>
        </div>

        {/* Learning Rate */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">Learning Rate</span>
          <select
            value={config.learningRate}
            onChange={(e) => updateConfig({ learningRate: parseFloat(e.target.value) })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-2.5 py-1.5 border border-zinc-700 focus:border-white focus:outline-none"
          >
            <option value="0.001">1e-3 (Head Only)</option>
            <option value="0.0003">3e-4 (AdamW Default)</option>
            <option value="0.0001">1e-4 (Fine-Tuning)</option>
            <option value="0.00001">1e-5 (Deep Backbone)</option>
          </select>
        </div>

        {/* Acceleration Device */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">Hardware Target</span>
          <select
            value={config.device}
            onChange={(e) => updateConfig({ device: e.target.value as DeviceType })}
            className="w-full bg-black text-white text-xs font-mono font-bold px-2.5 py-1.5 border border-zinc-700 focus:border-white focus:outline-none"
          >
            <option value="cuda">CUDA GPU</option>
            <option value="mps">Apple MPS</option>
            <option value="cpu">Standard CPU</option>
          </select>
        </div>

        {/* Mixed Precision */}
        <div className="space-y-1">
          <span className="text-[10px] text-zinc-400 font-mono uppercase font-black tracking-wider">FP16 / AMP</span>
          <button
            type="button"
            onClick={() => updateConfig({ useAmp: !config.useAmp })}
            className={`w-full py-1.5 px-2.5 text-xs font-mono font-black uppercase tracking-wider border transition cursor-pointer ${
              config.useAmp
                ? 'bg-yellow-400 text-black border-yellow-400'
                : 'bg-black text-zinc-400 border-zinc-700 hover:text-white'
            }`}
          >
            {config.useAmp ? 'AMP ON (FP16)' : 'FP32 FULL'}
          </button>
        </div>
      </div>
    </div>
  );
};

