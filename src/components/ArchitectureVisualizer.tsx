import React, { useState } from 'react';
import {
  Layers,
  ArrowRight,
  Zap,
  Filter,
  Minimize2,
  Share2,
  Flame,
  CheckCircle,
  HelpCircle
} from 'lucide-react';

interface Stage {
  id: string;
  number: string;
  name: string;
  subTitle: string;
  tensorShape: string;
  formula: string;
  explanation: string;
  keyConcepts: string[];
}

const STAGES: Stage[] = [
  {
    id: 'stage_input',
    number: '01',
    name: 'Image Input & Tensor Preprocessing',
    subTitle: 'Pixel Normalization and Channel Transformation',
    tensorShape: '(Batch, 3, 224, 224)',
    formula: 'x_{norm} = \\frac{x - \\mu_{ImageNet}}{\\sigma_{ImageNet}}',
    explanation:
      'Raw JPEG/PNG pixel arrays (0-255 uint8) are converted to 32-bit floating point tensors in RGB order and normalized to standard distribution (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).',
    keyConcepts: ['Channel Transpose (H,W,C -> C,H,W)', 'Zero-Centering Normalization', 'GPU Batch Tensors'],
  },
  {
    id: 'stage_conv',
    number: '02',
    name: '2D Convolution & Patch Embeddings',
    subTitle: 'Spatial Feature Extraction with Learnable Kernels',
    tensorShape: '(Batch, 64, 112, 112) -> (Batch, 512, 7, 7)',
    formula: 'y[i, j] = \\sum_{m} \\sum_{n} x[i+m, j+n] \\cdot W[m, n] + b',
    explanation:
      'Sliding 2D filters (e.g. 3x3 kernels) compute dot products across local pixel neighborhoods to extract hierarchical low-level features (edges, textures, corners) that progressively aggregate into high-level object concepts.',
    keyConcepts: ['Weight Sharing', 'Translation Equivariance', 'Receptive Field Expansion', 'Strided Pooling'],
  },
  {
    id: 'stage_bottleneck',
    number: '03',
    name: 'Residual Blocks / Self-Attention Heads',
    subTitle: 'Deep Non-Linear Representations & Skip Connections',
    tensorShape: '(Batch, 2048, 7, 7) or (Batch, 197, 768)',
    formula: 'y = \\mathcal{F}(x, \\{W_i\\}) + x \\quad \\text{or} \\quad \\text{Softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
    explanation:
      'Residual skip connections (x + F(x)) prevent gradient vanishing in deep 50-150 layer networks by providing an uninterrupted gradient highway. Vision Transformers instead compute pairwise token self-attention across the whole image.',
    keyConcepts: ['Residual Highway', 'Multi-Head Attention (MHA)', 'Batch/Layer Normalization', 'GELU / ReLU'],
  },
  {
    id: 'stage_pool',
    number: '04',
    name: 'Global Average Pooling (GAP)',
    subTitle: 'Spatial Dimension Collapse into Compact Feature Vector',
    tensorShape: '(Batch, 2048, 1, 1) -> (Batch, 2048)',
    formula: 'v_c = \\frac{1}{H \\times W} \\sum_{i=1}^H \\sum_{j=1}^W A_{c, i, j}',
    explanation:
      'Rather than flattening millions of parameters which causes overfitting, Global Average Pooling takes the spatial average of each 2D activation slice, yielding a clean invariant feature representation vector.',
    keyConcepts: ['Extreme Parameter Reduction', 'Spatial Translation Invariance', 'Overfitting Prevention'],
  },
  {
    id: 'stage_head',
    number: '05',
    name: 'Dense Linear Head & Softmax Probability',
    subTitle: 'Logits Projection into Class Probabilities',
    tensorShape: '(Batch, Num_Classes)',
    formula: 'P(y = k \\mid x) = \\frac{e^{z_k}}{\\sum_{j=1}^K e^{z_j}}',
    explanation:
      'A dense matrix multiplication projects the 2048-dimensional embedding into K class logits. The Softmax function normalizes logits into a valid probability distribution that strictly sums to 1.0 (100%).',
    keyConcepts: ['Cross-Entropy Loss (-log P)', 'Label Smoothing', 'Top-1 & Top-5 Argmax Prediction'],
  },
];

export const ArchitectureVisualizer: React.FC = () => {
  const [activeStageId, setActiveStageId] = useState<string>('stage_input');
  const activeStage = STAGES.find((s) => s.id === activeStageId) || STAGES[0];

  return (
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="bg-zinc-950 border-2 border-white p-6 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider bg-yellow-400 text-black border border-white">
                003 // ARCHITECTURE
              </span>
              <span className="text-xs text-zinc-300 font-mono font-bold uppercase">
                END-TO-END PIPELINE BREAKDOWN
              </span>
            </div>
            <h2 className="font-display text-xl sm:text-2xl font-black text-white uppercase tracking-tight mt-1">
              Neural Vision Computational Pipeline
            </h2>
            <p className="font-mono text-xs text-zinc-400 uppercase tracking-wide max-w-3xl mt-1">
              Select each sequential stage to inspect tensor shapes, mathematical formulations, and layer transformations.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Stage Pipeline Breadcrumbs */}
      <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
        {STAGES.map((stage) => {
          const isSelected = stage.id === activeStageId;
          return (
            <button
              key={stage.id}
              onClick={() => setActiveStageId(stage.id)}
              className={`p-4 text-left border-2 transition-all duration-150 cursor-pointer flex flex-col justify-between ${
                isSelected
                  ? 'bg-yellow-400 border-white text-black shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] font-black'
                  : 'bg-zinc-950 border-zinc-700 hover:border-white text-zinc-300'
              }`}
            >
              <div>
                <div className="flex items-center justify-between text-xs font-mono font-black mb-1.5">
                  <span className={isSelected ? 'text-black font-black' : 'text-yellow-400'}>
                    STAGE {stage.number}
                  </span>
                  {isSelected && <CheckCircle className="w-4 h-4 text-black stroke-[3]" />}
                </div>
                <div className={`font-display text-xs uppercase tracking-tight line-clamp-2 ${isSelected ? 'text-black font-black' : 'text-white font-bold'}`}>
                  {stage.name}
                </div>
              </div>
              <div className={`mt-3 font-mono text-[10px] font-bold px-2 py-1 border truncate ${
                isSelected ? 'bg-black text-yellow-400 border-black' : 'bg-black text-zinc-400 border-zinc-700'
              }`}>
                {stage.tensorShape}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Deep Dive Card */}
      <div className="bg-zinc-950 border-2 border-white p-6 shadow-[4px_4px_0px_0px_rgba(255,255,255,1)] space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b-2 border-zinc-800 pb-4">
          <div>
            <div className="text-xs font-mono font-black text-yellow-400 uppercase tracking-wider">
              STAGE {activeStage.number} // DEEP DIVE SPECIFICATION
            </div>
            <h3 className="font-display text-xl font-black text-white uppercase tracking-tight mt-0.5">
              {activeStage.name}
            </h3>
            <p className="font-mono text-xs text-zinc-400 uppercase">{activeStage.subTitle}</p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 font-mono font-black uppercase">TENSOR SHAPE:</span>
            <span className="px-3 py-1 bg-black text-yellow-400 border-2 border-yellow-400 text-xs font-mono font-black">
              {activeStage.tensorShape}
            </span>
          </div>
        </div>

        {/* Math & Conceptual Overview Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Explanation & Concepts */}
          <div className="space-y-4">
            <h4 className="font-display text-xs font-black text-white uppercase tracking-wider">
              Mechanism & Operation
            </h4>
            <p className="font-mono text-xs text-zinc-300 leading-relaxed bg-black p-4 border-2 border-zinc-700 uppercase">
              {activeStage.explanation}
            </p>

            <div className="space-y-2">
              <h5 className="font-mono text-xs font-black text-zinc-400 uppercase">Core Engineering Pillars:</h5>
              <div className="flex flex-wrap gap-2">
                {activeStage.keyConcepts.map((concept, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 text-xs font-mono font-bold uppercase bg-black text-white border border-white"
                  >
                    • {concept}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Mathematical Formulation & Code snippet */}
          <div className="space-y-4">
            <h4 className="font-display text-xs font-black text-white uppercase tracking-wider">
              Mathematical Formulation
            </h4>
            <div className="bg-black p-4 border-2 border-zinc-700 font-mono text-xs text-yellow-400 flex items-center justify-center min-h-[100px] text-center font-bold">
              <code>{activeStage.formula}</code>
            </div>

            <div className="bg-black p-4 border-2 border-zinc-800 space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-mono text-white font-black uppercase">
                <Zap className="w-3.5 h-3.5 text-yellow-400 stroke-[2.5]" />
                <span>PyTorch Module Equivalent:</span>
              </div>
              <pre className="text-xs font-mono text-zinc-300 overflow-x-auto p-2.5 bg-zinc-900 border border-zinc-700 font-bold">
                {activeStage.id === 'stage_input'
                  ? 'transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
                  : activeStage.id === 'stage_conv'
                  ? 'nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)'
                  : activeStage.id === 'stage_bottleneck'
                  ? 'out = self.conv2(self.relu(self.bn1(self.conv1(x)))) + identity_residual'
                  : activeStage.id === 'stage_pool'
                  ? 'nn.AdaptiveAvgPool2d((1, 1))  # (B, 2048, 7, 7) -> (B, 2048, 1, 1)'
                  : 'nn.Linear(2048, num_classes) -> torch.softmax(logits, dim=1)'}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
