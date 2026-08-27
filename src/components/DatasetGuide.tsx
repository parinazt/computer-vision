import React, { useState } from 'react';
import { generateDirectoryTree } from '../utils/codeGenerators';
import { FolderTree, Terminal, Sparkles, Copy, Check, CheckCircle2 } from 'lucide-react';

export const DatasetGuide: React.FC = () => {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-zinc-950 border-2 border-white p-6 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider bg-yellow-400 text-black border border-white">
            004 // DATA PREPARATION
          </span>
          <span className="text-xs text-zinc-300 font-mono font-bold uppercase">IMAGEFOLDER PROTOCOL</span>
        </div>
        <h2 className="font-display text-xl sm:text-2xl font-black text-white uppercase tracking-tight mt-1">
          Dataset Structure & Directory Conventions
        </h2>
        <p className="font-mono text-xs text-zinc-400 uppercase tracking-wide max-w-3xl mt-1">
          Standard layout for automatic ingestion by PyTorch torchvision.datasets.ImageFolder and TensorFlow tf.keras.utils.image_dataset_from_directory.
        </p>
      </div>

      {/* Grid: Folder Tree vs Shell Commands */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Directory Tree Structure */}
        <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
          <div className="flex items-center justify-between border-b-2 border-zinc-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-yellow-400 text-black font-black flex items-center justify-center border border-white">
                <FolderTree className="w-3.5 h-3.5 stroke-[2.5]" />
              </div>
              <h3 className="font-display text-sm font-black text-white uppercase tracking-tight">
                Recommended Folder Hierarchy
              </h3>
            </div>
            <span className="text-[10px] font-mono font-black uppercase bg-white text-black px-2 py-0.5 border border-white">
              STANDARD
            </span>
          </div>

          <p className="font-mono text-xs text-zinc-400 uppercase">
            Subfolder directory names directly become target class labels. Split datasets into train and validation sets (e.g. 80/20 ratio).
          </p>

          <div className="bg-black p-4 border-2 border-zinc-700 font-mono text-xs text-yellow-400 leading-relaxed overflow-x-auto font-bold">
            <pre>{generateDirectoryTree()}</pre>
          </div>

          <div className="p-3 bg-black border-2 border-zinc-800 text-xs font-mono text-zinc-300 space-y-1.5">
            <div className="flex items-center gap-1.5 text-white font-black uppercase">
              <CheckCircle2 className="w-3.5 h-3.5 text-yellow-400 stroke-[2.5]" />
              <span>SUPPORTED IMAGE FORMATS:</span>
            </div>
            <p className="text-[11px] text-zinc-400 uppercase">
              JPEG (.jpg, .jpeg), PNG (.png), WebP (.webp), BMP (.bmp).
            </p>
          </div>
        </div>

        {/* Right: Quick Terminal Setup Commands */}
        <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
          <div className="flex items-center justify-between border-b-2 border-zinc-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-white text-black font-black flex items-center justify-center border border-white">
                <Terminal className="w-3.5 h-3.5 stroke-[2.5]" />
              </div>
              <h3 className="font-display text-sm font-black text-white uppercase tracking-tight">
                CLI Environment & Setup Commands
              </h3>
            </div>
            <span className="text-[10px] font-mono font-black uppercase bg-yellow-400 text-black px-2 py-0.5 border border-white">
              BASH / CONDA
            </span>
          </div>

          <div className="space-y-3 font-mono">
            {/* Command 1: PyTorch install */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-black text-white uppercase">1. Install PyTorch with CUDA 12.1:</span>
                <button
                  onClick={() => copyToClipboard('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121', 'cmd1')}
                  className="text-yellow-400 hover:text-white text-[11px] font-black uppercase flex items-center gap-1 cursor-pointer border border-yellow-400 px-2 py-0.5 bg-black"
                >
                  {copiedCmd === 'cmd1' ? <Check className="w-3 h-3 text-white stroke-[3]" /> : <Copy className="w-3 h-3" />}
                  {copiedCmd === 'cmd1' ? 'COPIED' : 'COPY'}
                </button>
              </div>
              <div className="bg-black p-2.5 border border-zinc-700 font-mono text-xs text-zinc-300 overflow-x-auto font-bold">
                <code>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</code>
              </div>
            </div>

            {/* Command 2: Helper packages */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-black text-white uppercase">2. Install Computer Vision Stack:</span>
                <button
                  onClick={() => copyToClipboard('pip install opencv-python pillow matplotlib scikit-learn tqdm', 'cmd2')}
                  className="text-yellow-400 hover:text-white text-[11px] font-black uppercase flex items-center gap-1 cursor-pointer border border-yellow-400 px-2 py-0.5 bg-black"
                >
                  {copiedCmd === 'cmd2' ? <Check className="w-3 h-3 text-white stroke-[3]" /> : <Copy className="w-3 h-3" />}
                  {copiedCmd === 'cmd2' ? 'COPIED' : 'COPY'}
                </button>
              </div>
              <div className="bg-black p-2.5 border border-zinc-700 font-mono text-xs text-zinc-300 overflow-x-auto font-bold">
                <code>pip install opencv-python pillow matplotlib scikit-learn tqdm</code>
              </div>
            </div>

            {/* Command 3: Automatic Folder Scaffolding */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-black text-white uppercase">3. Scaffold Directory Layout:</span>
                <button
                  onClick={() => copyToClipboard('mkdir -p data/train/class_{1,2,3} data/val/class_{1,2,3}', 'cmd3')}
                  className="text-yellow-400 hover:text-white text-[11px] font-black uppercase flex items-center gap-1 cursor-pointer border border-yellow-400 px-2 py-0.5 bg-black"
                >
                  {copiedCmd === 'cmd3' ? <Check className="w-3 h-3 text-white stroke-[3]" /> : <Copy className="w-3 h-3" />}
                  {copiedCmd === 'cmd3' ? 'COPIED' : 'COPY'}
                </button>
              </div>
              <div className="bg-black p-2.5 border border-zinc-700 font-mono text-xs text-zinc-300 overflow-x-auto font-bold">
                <code>mkdir -p data/train/class_{'{1,2,3}'} data/val/class_{'{1,2,3}'}</code>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Best Practices Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
        <div className="bg-zinc-950 border-2 border-white p-4 space-y-2 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
          <div className="flex items-center gap-2 text-yellow-400 font-black text-xs uppercase">
            <Sparkles className="w-4 h-4 stroke-[2.5]" />
            <span>1. DATA AUGMENTATION</span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed uppercase">
            Apply RandomHorizontalFlip and ColorJitter exclusively to train sets. Never augment validation or test partitions beyond standard resize & normalize.
          </p>
        </div>

        <div className="bg-zinc-950 border-2 border-white p-4 space-y-2 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
          <div className="flex items-center gap-2 text-white font-black text-xs uppercase">
            <Sparkles className="w-4 h-4 stroke-[2.5]" />
            <span>2. TRANSFER WARMUP</span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed uppercase">
            Freeze backbone feature extractors for the first 3-5 warmup epochs while training the dense classifier head, then unfreeze with lower learning rate.
          </p>
        </div>

        <div className="bg-zinc-950 border-2 border-white p-4 space-y-2 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
          <div className="flex items-center gap-2 text-yellow-400 font-black text-xs uppercase">
            <Sparkles className="w-4 h-4 stroke-[2.5]" />
            <span>3. CLASS IMBALANCE</span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed uppercase">
            For skewed class distributions, compute inverse frequency loss weights and configure nn.CrossEntropyLoss(weight=class_weights).
          </p>
        </div>
      </div>

      {/* GitHub Repository & Python Scripts Section */}
      <div className="bg-zinc-950 border-2 border-white p-5 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-4">
        <div className="flex items-center justify-between border-b-2 border-zinc-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-white text-black font-black flex items-center justify-center border border-white">
              <Terminal className="w-3.5 h-3.5 stroke-[2.5]" />
            </div>
            <h3 className="font-display text-sm font-black text-white uppercase tracking-tight">
              Publishing Python Code & Scripts to GitHub
            </h3>
          </div>
          <span className="text-[10px] font-mono font-black uppercase bg-yellow-400 text-black px-2 py-0.5 border border-white">
            READY IN WORKSPACE
          </span>
        </div>

        <p className="font-mono text-xs text-zinc-300 uppercase leading-relaxed">
          The workspace now includes pre-configured Python scripts in <code className="text-yellow-400">/scripts</code> and <code className="text-yellow-400">requirements.txt</code>:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
          <div className="p-3 bg-black border border-zinc-700 space-y-1">
            <div className="text-yellow-400 font-black">scripts/train_pytorch.py</div>
            <p className="text-zinc-400 text-[11px] uppercase">
              PyTorch ImageFolder model trainer with Mixed Precision (AMP), Cosine Annealing, and checkpoint saves.
            </p>
          </div>
          <div className="p-3 bg-black border border-zinc-700 space-y-1">
            <div className="text-white font-black">scripts/train_tensorflow.py</div>
            <p className="text-zinc-400 text-[11px] uppercase">
              TensorFlow Keras pipeline utilizing EfficientNet backbone with early stopping and learning rate plateau scheduling.
            </p>
          </div>
          <div className="p-3 bg-black border border-zinc-700 space-y-1">
            <div className="text-yellow-400 font-black">scripts/inference_webcam.py</div>
            <p className="text-zinc-400 text-[11px] uppercase">
              Real-time video feed inference with OpenCV and PyTorch, displaying live bounding boxes and FPS telemetry.
            </p>
          </div>
        </div>

        <div className="p-3 bg-black border-2 border-zinc-800 font-mono text-xs space-y-2">
          <div className="text-white font-black uppercase flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-yellow-400 stroke-[2.5]" />
            <span>How to push this workspace to GitHub:</span>
          </div>
          <ol className="list-decimal list-inside space-y-1 text-zinc-300 text-[11px] uppercase">
            <li>In AI Studio, click the project menu / settings icon in the top header and select <span className="text-yellow-400 font-bold">Export to GitHub</span> or <span className="text-yellow-400 font-bold">Export ZIP</span>.</li>
            <li>Alternatively, run <code className="text-white bg-zinc-900 px-1 py-0.5 border border-zinc-700">git add scripts/ requirements.txt README.md && git commit -m "Add vision scripts" && git push</code> in your local clone.</li>
          </ol>
        </div>
      </div>
    </div>
  );
};
