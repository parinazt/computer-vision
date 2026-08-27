import React, { useState, useMemo } from 'react';
import {
  CodeGeneratorConfig,
  Framework,
  ModelBackbone,
} from './types';
import { generatePythonCode, generateRequirementsTxt } from './utils/codeGenerators';
import { CodeViewer } from './components/CodeViewer';
import { ConfiguratorPanel } from './components/ConfiguratorPanel';
import { ClassifierPlayground } from './components/ClassifierPlayground';
import { ArchitectureVisualizer } from './components/ArchitectureVisualizer';
import { DatasetGuide } from './components/DatasetGuide';
import {
  Code2,
  Eye,
  Layers,
  FolderTree,
  Terminal,
  Sparkles,
  Zap,
  CheckCircle2,
  Download,
  BookOpen,
  Cpu,
  Boxes
} from 'lucide-react';

const DEFAULT_CONFIG: CodeGeneratorConfig = {
  framework: 'pytorch',
  taskType: 'transfer_learning',
  modelBackbone: 'resnet50',
  dataset: 'custom_folder',
  numClasses: 5,
  imageSize: 224,
  batchSize: 32,
  epochs: 20,
  learningRate: 0.0003,
  optimizer: 'AdamW',
  weightDecay: 0.0001,
  useAmp: true,
  pretrained: true,
  freezeBackbone: false,
  device: 'cuda',
  includeAugmentation: true,
  savePath: 'best_vision_model.pth',
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'generator' | 'playground' | 'architecture' | 'dataset'>('generator');
  const [config, setConfig] = useState<CodeGeneratorConfig>(DEFAULT_CONFIG);
  const [activeView, setActiveView] = useState<'code' | 'requirements'>('code');

  const generatedPythonCode = useMemo(() => {
    return generatePythonCode(config);
  }, [config]);

  const generatedRequirements = useMemo(() => {
    return generateRequirementsTxt(config.framework);
  }, [config.framework]);

  const setPreset = (presetFw: Framework, backbone: ModelBackbone, task = 'transfer_learning') => {
    setConfig((prev) => ({
      ...prev,
      framework: presetFw,
      modelBackbone: backbone,
      taskType: task as any,
    }));
    setActiveTab('generator');
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-yellow-400 selection:text-black pb-20">
      {/* Top Header with Bold Stark Typography */}
      <header className="sticky top-0 z-50 bg-black/95 backdrop-blur-md border-b-2 border-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-yellow-400 text-black flex items-center justify-center font-black border-2 border-white shadow-[2px_2px_0px_0px_rgba(255,255,255,1)]">
              <Eye className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-display text-lg sm:text-xl font-black uppercase tracking-tight text-white">
                  Vision Classifier <span className="text-yellow-400">Studio</span>
                </h1>
                <span className="px-2 py-0.5 text-[10px] font-black uppercase tracking-wider bg-white text-black border border-white">
                  CV ENGINE
                </span>
              </div>
              <p className="text-[11px] font-mono text-zinc-400 uppercase tracking-wider hidden sm:block">
                001 // Deep Learning Image Classification Code Generator
              </p>
            </div>
          </div>

          {/* Primary Navigation Tabs with Bold Box Style */}
          <nav className="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto pb-1 md:pb-0">
            <button
              onClick={() => setActiveTab('generator')}
              className={`flex items-center gap-2 px-3.5 py-2 text-xs font-black uppercase tracking-wider transition-all cursor-pointer border-2 ${
                activeTab === 'generator'
                  ? 'bg-yellow-400 text-black border-white shadow-[3px_3px_0px_0px_rgba(255,255,255,1)]'
                  : 'bg-zinc-950 text-zinc-300 border-zinc-700 hover:border-white hover:text-white'
              }`}
            >
              <Code2 className="w-4 h-4 stroke-[2.5]" />
              <span>Python Scripts</span>
            </button>

            <button
              onClick={() => setActiveTab('playground')}
              className={`flex items-center gap-2 px-3.5 py-2 text-xs font-black uppercase tracking-wider transition-all cursor-pointer border-2 ${
                activeTab === 'playground'
                  ? 'bg-yellow-400 text-black border-white shadow-[3px_3px_0px_0px_rgba(255,255,255,1)]'
                  : 'bg-zinc-950 text-zinc-300 border-zinc-700 hover:border-white hover:text-white'
              }`}
            >
              <Eye className="w-4 h-4 stroke-[2.5]" />
              <span>Vision Playground</span>
            </button>

            <button
              onClick={() => setActiveTab('architecture')}
              className={`flex items-center gap-2 px-3.5 py-2 text-xs font-black uppercase tracking-wider transition-all cursor-pointer border-2 ${
                activeTab === 'architecture'
                  ? 'bg-yellow-400 text-black border-white shadow-[3px_3px_0px_0px_rgba(255,255,255,1)]'
                  : 'bg-zinc-950 text-zinc-300 border-zinc-700 hover:border-white hover:text-white'
              }`}
            >
              <Layers className="w-4 h-4 stroke-[2.5]" />
              <span>Architecture</span>
            </button>

            <button
              onClick={() => setActiveTab('dataset')}
              className={`flex items-center gap-2 px-3.5 py-2 text-xs font-black uppercase tracking-wider transition-all cursor-pointer border-2 ${
                activeTab === 'dataset'
                  ? 'bg-yellow-400 text-black border-white shadow-[3px_3px_0px_0px_rgba(255,255,255,1)]'
                  : 'bg-zinc-950 text-zinc-300 border-zinc-700 hover:border-white hover:text-white'
              }`}
            >
              <FolderTree className="w-4 h-4 stroke-[2.5]" />
              <span>Dataset Setup</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        {/* Quick Recipe Preset Bar */}
        <div className="bg-zinc-950 border-2 border-zinc-800 p-3 flex items-center gap-2.5 overflow-x-auto scrollbar-none">
          <span className="font-mono text-xs font-black text-yellow-400 uppercase tracking-wider whitespace-nowrap flex items-center gap-1.5 pr-2 border-r-2 border-zinc-800">
            <Zap className="w-4 h-4 text-yellow-400 stroke-[2.5]" /> PRESETS:
          </span>
          <button
            onClick={() => setPreset('pytorch', 'resnet50', 'transfer_learning')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.framework === 'pytorch' && config.modelBackbone === 'resnet50' && config.taskType === 'transfer_learning'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            PyTorch ResNet-50
          </button>
          <button
            onClick={() => setPreset('pytorch', 'vit_b_16', 'transfer_learning')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.framework === 'pytorch' && config.modelBackbone === 'vit_b_16'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            Vision Transformer (ViT-B/16)
          </button>
          <button
            onClick={() => setPreset('tensorflow', 'mobilenet_v3_large', 'transfer_learning')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.framework === 'tensorflow'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            TensorFlow / MobileNet-V3
          </button>
          <button
            onClick={() => setPreset('pytorch', 'custom_cnn_4layer', 'custom_cnn')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.taskType === 'custom_cnn'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            From-Scratch 4-Layer CNN
          </button>
          <button
            onClick={() => setPreset('opencv_webcam', 'mobilenet_v3_large', 'inference_only')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.framework === 'opencv_webcam'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            Real-Time Webcam CV
          </button>
          <button
            onClick={() => setPreset('sklearn', 'resnet50', 'transfer_learning')}
            className={`px-3 py-1.5 border text-xs font-bold uppercase tracking-wider whitespace-nowrap transition cursor-pointer ${
              config.framework === 'sklearn'
                ? 'bg-white text-black border-white font-black'
                : 'bg-black text-zinc-300 border-zinc-700 hover:border-zinc-400 hover:text-white'
            }`}
          >
            OpenCV HOG + SVM
          </button>
        </div>

        {/* Tab 1: Python Code Generator */}
        {activeTab === 'generator' && (
          <div className="space-y-6">
            {/* Interactive Pipeline Parameters Panel */}
            <ConfiguratorPanel
              config={config}
              onChange={setConfig}
              activeView={activeView}
              setActiveView={setActiveView}
            />

            {/* Generated Code Display */}
            {activeView === 'code' ? (
              <CodeViewer
                code={generatedPythonCode}
                filename={
                  config.framework === 'opencv_webcam'
                    ? 'webcam_classifier.py'
                    : config.taskType === 'custom_cnn'
                    ? 'custom_cnn_classifier.py'
                    : config.taskType === 'inference_only'
                    ? 'inference.py'
                    : 'train_classifier.py'
                }
                title={`${config.framework.toUpperCase()} • ${config.modelBackbone.toUpperCase()} (${config.numClasses} classes)`}
              />
            ) : (
              <CodeViewer
                code={generatedRequirements}
                filename="requirements.txt"
                language="python"
                title="Python Dependencies"
              />
            )}
          </div>
        )}

        {/* Tab 2: Vision Classifier Playground */}
        {activeTab === 'playground' && <ClassifierPlayground />}

        {/* Tab 3: Architecture Deep-Dive */}
        {activeTab === 'architecture' && <ArchitectureVisualizer />}

        {/* Tab 4: Dataset & Directory Setup */}
        {activeTab === 'dataset' && <DatasetGuide />}
      </main>
    </div>
  );
}
